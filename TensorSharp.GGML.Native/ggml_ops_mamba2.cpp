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

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

using namespace tsg;

namespace
{
    struct Mamba2PrefillCacheKey
    {
        int T;
        int d_in_proj_total;
        int d_inner;
        int d_state;
        int n_head;
        int head_dim;
        int n_group;
        int d_conv;
        bool has_conv_bias;
        bool has_d;
        bool has_norm;
        std::uintptr_t projected_ptr;
        std::uintptr_t hidden_out_ptr;
        float eps;
    };

    struct Mamba2PrefillCacheKeyEq
    {
        bool operator()(const Mamba2PrefillCacheKey& a, const Mamba2PrefillCacheKey& b) const noexcept
        {
            return a.T == b.T &&
                a.d_in_proj_total == b.d_in_proj_total &&
                a.d_inner == b.d_inner &&
                a.d_state == b.d_state &&
                a.n_head == b.n_head &&
                a.head_dim == b.head_dim &&
                a.n_group == b.n_group &&
                a.d_conv == b.d_conv &&
                a.has_conv_bias == b.has_conv_bias &&
                a.has_d == b.has_d &&
                a.has_norm == b.has_norm &&
                a.projected_ptr == b.projected_ptr &&
                a.hidden_out_ptr == b.hidden_out_ptr &&
                a.eps == b.eps;
        }
    };

    struct Mamba2PrefillCacheKeyHash
    {
        std::size_t operator()(const Mamba2PrefillCacheKey& k) const noexcept
        {
            std::size_t h = static_cast<std::size_t>(k.T);
            h = h * 1315423911u + static_cast<std::size_t>(k.d_in_proj_total);
            h = h * 1315423911u + static_cast<std::size_t>(k.d_inner);
            h = h * 1315423911u + static_cast<std::size_t>(k.d_state);
            h = h * 1315423911u + static_cast<std::size_t>(k.n_head);
            h = h * 1315423911u + static_cast<std::size_t>(k.head_dim);
            h = h * 1315423911u + static_cast<std::size_t>(k.n_group);
            h = h * 1315423911u + static_cast<std::size_t>(k.d_conv);
            h = h * 1315423911u + static_cast<std::size_t>(k.has_conv_bias ? 1 : 0);
            h = h * 1315423911u + static_cast<std::size_t>(k.has_d ? 1 : 0);
            h = h * 1315423911u + static_cast<std::size_t>(k.has_norm ? 1 : 0);
            h = h * 1315423911u + static_cast<std::size_t>(k.projected_ptr);
            h = h * 1315423911u + static_cast<std::size_t>(k.hidden_out_ptr);
            std::uint32_t eps_bits;
            std::memcpy(&eps_bits, &k.eps, sizeof(eps_bits));
            h = h * 1315423911u + static_cast<std::size_t>(eps_bits);
            return h;
        }
    };

    struct Mamba2PrefillCacheEntry
    {
        std::unique_ptr<std::uint8_t[]> ctx_buffer;
        ggml_context* ctx = nullptr;
        BufferHandle buffer{nullptr};
        BufferHandle projected_host_buffer{nullptr};
        BufferHandle hidden_out_host_buffer{nullptr};
        ggml_cgraph* graph = nullptr;

        ggml_tensor* projected_storage = nullptr;
        ggml_tensor* hidden_out_storage = nullptr;
        ggml_tensor* conv_state_storage = nullptr;
        ggml_tensor* ssm_state_storage = nullptr;
        ggml_tensor* conv_weight_storage = nullptr;
        ggml_tensor* conv_bias_storage = nullptr;
        ggml_tensor* dt_bias_storage = nullptr;
        ggml_tensor* a_storage = nullptr;
        ggml_tensor* d_storage = nullptr;
        ggml_tensor* norm_storage = nullptr;
        ggml_tensor* ids_storage = nullptr;

        std::size_t projected_bytes = 0;
        std::size_t hidden_out_bytes = 0;
        std::size_t conv_state_bytes = 0;
        std::size_t ssm_state_bytes = 0;
        std::size_t conv_weight_bytes = 0;
        std::size_t conv_bias_bytes = 0;
        std::size_t dt_bias_bytes = 0;
        std::size_t a_bytes = 0;
        std::size_t d_bytes = 0;
        std::size_t norm_bytes = 0;
        bool projected_zero_copy = false;
        bool hidden_out_zero_copy = false;

        std::mutex compute_mutex;

        ~Mamba2PrefillCacheEntry()
        {
            if (ctx != nullptr)
            {
                ggml_free(ctx);
                ctx = nullptr;
            }
        }
    };

    struct Mamba2DecodeCacheKey
    {
        std::uint64_t state_key;
        int d_in_proj_total;
        int d_inner;
        int d_state;
        int n_head;
        int head_dim;
        int n_group;
        int d_conv;
        bool has_conv_bias;
        bool has_d;
        bool has_norm;
        std::uintptr_t projected_ptr;
        std::uintptr_t hidden_out_ptr;
        float eps;
    };

    struct Mamba2DecodeCacheKeyEq
    {
        bool operator()(const Mamba2DecodeCacheKey& a, const Mamba2DecodeCacheKey& b) const noexcept
        {
            return a.state_key == b.state_key &&
                a.d_in_proj_total == b.d_in_proj_total &&
                a.d_inner == b.d_inner &&
                a.d_state == b.d_state &&
                a.n_head == b.n_head &&
                a.head_dim == b.head_dim &&
                a.n_group == b.n_group &&
                a.d_conv == b.d_conv &&
                a.has_conv_bias == b.has_conv_bias &&
                a.has_d == b.has_d &&
                a.has_norm == b.has_norm &&
                a.projected_ptr == b.projected_ptr &&
                a.hidden_out_ptr == b.hidden_out_ptr &&
                a.eps == b.eps;
        }
    };

    struct Mamba2DecodeCacheKeyHash
    {
        std::size_t operator()(const Mamba2DecodeCacheKey& k) const noexcept
        {
            std::size_t h = static_cast<std::size_t>(k.state_key);
            h = h * 1315423911u + static_cast<std::size_t>(k.d_in_proj_total);
            h = h * 1315423911u + static_cast<std::size_t>(k.d_inner);
            h = h * 1315423911u + static_cast<std::size_t>(k.d_state);
            h = h * 1315423911u + static_cast<std::size_t>(k.n_head);
            h = h * 1315423911u + static_cast<std::size_t>(k.head_dim);
            h = h * 1315423911u + static_cast<std::size_t>(k.n_group);
            h = h * 1315423911u + static_cast<std::size_t>(k.d_conv);
            h = h * 1315423911u + static_cast<std::size_t>(k.has_conv_bias ? 1 : 0);
            h = h * 1315423911u + static_cast<std::size_t>(k.has_d ? 1 : 0);
            h = h * 1315423911u + static_cast<std::size_t>(k.has_norm ? 1 : 0);
            h = h * 1315423911u + static_cast<std::size_t>(k.projected_ptr);
            h = h * 1315423911u + static_cast<std::size_t>(k.hidden_out_ptr);
            std::uint32_t eps_bits;
            std::memcpy(&eps_bits, &k.eps, sizeof(eps_bits));
            h = h * 1315423911u + static_cast<std::size_t>(eps_bits);
            return h;
        }
    };

    struct Mamba2DecodeCacheEntry
    {
        std::unique_ptr<std::uint8_t[]> ctx_buffer;
        ggml_context* ctx = nullptr;
        BufferHandle buffer{nullptr};
        BufferHandle projected_host_buffer{nullptr};
        BufferHandle hidden_out_host_buffer{nullptr};
        ggml_cgraph* graph = nullptr;

        ggml_tensor* projected_storage = nullptr;
        ggml_tensor* hidden_out_storage = nullptr;
        ggml_tensor* conv_state_storage = nullptr;
        ggml_tensor* ssm_state_storage = nullptr;
        ggml_tensor* conv_weight_storage = nullptr;
        ggml_tensor* conv_bias_storage = nullptr;
        ggml_tensor* dt_bias_storage = nullptr;
        ggml_tensor* a_storage = nullptr;
        ggml_tensor* d_storage = nullptr;
        ggml_tensor* norm_storage = nullptr;
        ggml_tensor* ids_storage = nullptr;

        std::size_t projected_bytes = 0;
        std::size_t hidden_out_bytes = 0;
        std::size_t conv_state_bytes = 0;
        std::size_t ssm_state_bytes = 0;
        std::size_t conv_weight_bytes = 0;
        std::size_t conv_bias_bytes = 0;
        std::size_t dt_bias_bytes = 0;
        std::size_t a_bytes = 0;
        std::size_t d_bytes = 0;
        std::size_t norm_bytes = 0;
        bool projected_zero_copy = false;
        bool hidden_out_zero_copy = false;
        bool weights_uploaded = false;
        bool state_initialized = false;

        std::mutex compute_mutex;

        ~Mamba2DecodeCacheEntry()
        {
            if (ctx != nullptr)
            {
                ggml_free(ctx);
                ctx = nullptr;
            }
        }
    };

    std::mutex g_mamba2_prefill_cache_mutex;
    std::unordered_map<Mamba2PrefillCacheKey,
                       std::unique_ptr<Mamba2PrefillCacheEntry>,
                       Mamba2PrefillCacheKeyHash,
                       Mamba2PrefillCacheKeyEq> g_mamba2_prefill_cache;

    std::mutex g_mamba2_decode_cache_mutex;
    std::unordered_map<Mamba2DecodeCacheKey,
                       std::unique_ptr<Mamba2DecodeCacheEntry>,
                       Mamba2DecodeCacheKeyHash,
                       Mamba2DecodeCacheKeyEq> g_mamba2_decode_cache;

    void ensure_mamba2_prefill_cache_cleanup_registered()
    {
        static std::once_flag flag;
        std::call_once(flag, []() {
            std::atexit([]() {
                std::lock_guard<std::mutex> lk(g_mamba2_prefill_cache_mutex);
                g_mamba2_prefill_cache.clear();
                std::lock_guard<std::mutex> lk_decode(g_mamba2_decode_cache_mutex);
                g_mamba2_decode_cache.clear();
            });
        });
    }

    bool validate_mamba2_prefill_args(
        const TensorView2DDesc& projected_desc,
        const TensorView2DDesc& hidden_out_desc,
        const void* conv_state_data,
        int conv_state_elements,
        const void* ssm_state_data,
        int ssm_state_elements,
        const void* conv_weight_data,
        const void* dt_bias_data,
        const void* a_data,
        int d_inner,
        int d_state,
        int n_head,
        int head_dim,
        int n_group,
        int d_conv)
    {
        if (!ensure_backend())
            return false;
        if (!validate_desc(projected_desc, "projected") ||
            !validate_desc(hidden_out_desc, "hidden_out"))
        {
            return false;
        }
        if (!can_map_standard_view(projected_desc) || !can_map_standard_view(hidden_out_desc))
        {
            set_last_error("NemotronMamba2Prefill: projected and hidden_out must be row-contiguous Float32 matrices.");
            return false;
        }
        if (d_inner <= 0 || d_state <= 0 || n_head <= 0 || head_dim <= 0 || n_group <= 0 || d_conv <= 0)
        {
            set_last_error("NemotronMamba2Prefill: invalid Mamba2 dimensions.");
            return false;
        }
        if (n_head * head_dim != d_inner)
        {
            set_last_error("NemotronMamba2Prefill: n_head * head_dim must equal d_inner.");
            return false;
        }
        if ((n_head % n_group) != 0 || (d_inner % n_group) != 0)
        {
            set_last_error("NemotronMamba2Prefill: n_head and d_inner must be divisible by n_group.");
            return false;
        }

        const int T = hidden_out_desc.dim0;
        const int xbc_size = d_inner + 2 * n_group * d_state;
        const int d_in_proj_total = 2 * d_inner + 2 * n_group * d_state + n_head;
        const int conv_dim = d_conv - 1;

        if (T <= 0 || projected_desc.dim0 != T ||
            projected_desc.dim1 != d_in_proj_total ||
            hidden_out_desc.dim1 != d_inner)
        {
            set_last_error("NemotronMamba2Prefill: projected/hidden_out shape mismatch.");
            return false;
        }
        if (conv_dim > 0 && conv_state_data == nullptr)
        {
            set_last_error("NemotronMamba2Prefill: conv_state must be non-null when d_conv > 1.");
            return false;
        }
        if (ssm_state_data == nullptr || conv_weight_data == nullptr || dt_bias_data == nullptr || a_data == nullptr)
        {
            set_last_error("NemotronMamba2Prefill: conv_weight, dt_bias, A, and ssm_state must be non-null.");
            return false;
        }
        if (conv_state_elements != conv_dim * xbc_size)
        {
            set_last_error("NemotronMamba2Prefill: conv_state length mismatch.");
            return false;
        }
        if (ssm_state_elements != d_state * head_dim * n_head)
        {
            set_last_error("NemotronMamba2Prefill: ssm_state length mismatch.");
            return false;
        }

        return true;
    }

    void update_conv_state_from_projected(
        float* conv_state,
        const TensorView2DDesc& projected_desc,
        int d_inner,
        int xbc_size,
        int conv_dim)
    {
        if (conv_dim <= 0 || conv_state == nullptr)
            return;

        const int T = projected_desc.dim0;
        const float* projected = static_cast<const float*>(projected_desc.data);

        auto copy_xbc_row = [&](int token, float* dst) {
            const float* row = projected + static_cast<std::size_t>(token) * projected_desc.stride0;
            const float* xbc = row + static_cast<std::size_t>(d_inner) * projected_desc.stride1;
            if (projected_desc.stride1 == 1)
            {
                std::memcpy(dst, xbc, static_cast<std::size_t>(xbc_size) * sizeof(float));
            }
            else
            {
                for (int i = 0; i < xbc_size; ++i)
                    dst[i] = xbc[static_cast<std::size_t>(i) * projected_desc.stride1];
            }
        };

        if (T >= conv_dim)
        {
            const int first_token = T - conv_dim;
            for (int k = 0; k < conv_dim; ++k)
                copy_xbc_row(first_token + k, conv_state + static_cast<std::size_t>(k) * xbc_size);
            return;
        }

        const int keep = conv_dim - T;
        if (keep > 0)
        {
            std::memmove(
                conv_state,
                conv_state + static_cast<std::size_t>(T) * xbc_size,
                static_cast<std::size_t>(keep) * xbc_size * sizeof(float));
        }

        for (int k = 0; k < T; ++k)
            copy_xbc_row(k, conv_state + static_cast<std::size_t>(keep + k) * xbc_size);
    }
}

TSG_EXPORT int TSGgml_NemotronMamba2PrefillF32(
    TensorView2DDesc projected_desc,
    TensorView2DDesc hidden_out_desc,
    void* conv_state_data,
    int conv_state_elements,
    void* ssm_state_data,
    int ssm_state_elements,
    void* conv_weight_data,
    void* conv_bias_data,
    void* dt_bias_data,
    void* a_data,
    void* d_data,
    void* ssm_norm_data,
    int d_inner,
    int d_state,
    int n_head,
    int head_dim,
    int n_group,
    int d_conv,
    float eps)
{
    try
    {
        if (!validate_mamba2_prefill_args(
                projected_desc, hidden_out_desc,
                conv_state_data, conv_state_elements,
                ssm_state_data, ssm_state_elements,
                conv_weight_data, dt_bias_data, a_data,
                d_inner, d_state, n_head, head_dim, n_group, d_conv))
        {
            return 0;
        }

        const int T = hidden_out_desc.dim0;
        const int conv_dim = d_conv - 1;
        const int xbc_size = d_inner + 2 * n_group * d_state;
        const int d_in_proj_total = 2 * d_inner + 2 * n_group * d_state + n_head;
        const int inner_per_group = d_inner / n_group;
        const bool has_conv_bias = conv_bias_data != nullptr;
        const bool has_d = d_data != nullptr;
        const bool has_norm = ssm_norm_data != nullptr;
        const std::size_t projected_bytes = static_cast<std::size_t>(projected_desc.raw_bytes);
        const std::size_t hidden_out_bytes = static_cast<std::size_t>(hidden_out_desc.raw_bytes);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        const bool projected_can_zero_copy =
            g_backend_type == BACKEND_TYPE_METAL &&
            projected_bytes >= 4096 &&
            can_use_host_ptr_buffer(g_backend, dev, projected_desc.data, projected_bytes);
        const bool hidden_out_can_zero_copy =
            g_backend_type == BACKEND_TYPE_METAL &&
            hidden_out_bytes >= 4096 &&
            can_use_host_ptr_buffer(g_backend, dev, hidden_out_desc.data, hidden_out_bytes);

        Mamba2PrefillCacheKey cache_key{
            T, d_in_proj_total, d_inner, d_state, n_head, head_dim, n_group, d_conv,
            has_conv_bias, has_d, has_norm,
            projected_can_zero_copy ? reinterpret_cast<std::uintptr_t>(projected_desc.data) : 0u,
            hidden_out_can_zero_copy ? reinterpret_cast<std::uintptr_t>(hidden_out_desc.data) : 0u,
            eps
        };

        Mamba2PrefillCacheEntry* entry = nullptr;
        {
            std::lock_guard<std::mutex> lk(g_mamba2_prefill_cache_mutex);
            auto it = g_mamba2_prefill_cache.find(cache_key);
            if (it != g_mamba2_prefill_cache.end())
                entry = it->second.get();
        }

        if (entry == nullptr)
        {
            const std::size_t per_tensor_bytes = 384;
            std::size_t ctx_size = 160 * per_tensor_bytes;
            if (ctx_size < 4 * 1024 * 1024) ctx_size = 4 * 1024 * 1024;

            auto new_entry = std::make_unique<Mamba2PrefillCacheEntry>();
            new_entry->ctx_buffer = std::make_unique<std::uint8_t[]>(ctx_size);

            ggml_init_params params = {};
            params.mem_size = ctx_size;
            params.mem_buffer = new_entry->ctx_buffer.get();
            params.no_alloc = true;
            new_entry->ctx = ggml_init(params);
            if (new_entry->ctx == nullptr)
            {
                set_last_error("NemotronMamba2Prefill: cached context init failed.");
                return 0;
            }

            ggml_context* ctx = new_entry->ctx;

            TensorBinding projected_bind{};
            TensorBinding hidden_out_bind{};

            if (projected_can_zero_copy)
            {
                ggml_backend_buffer_t projected_buffer = nullptr;
                if (create_binding_from_host_ptr_2d(ctx, g_backend, projected_desc, projected_bind, projected_buffer))
                {
                    new_entry->projected_host_buffer = BufferHandle(projected_buffer);
                    new_entry->projected_zero_copy = true;
                }
            }
            if (projected_bind.storage == nullptr)
            {
                projected_bind = create_standard_binding(ctx, projected_desc);
            }

            if (hidden_out_can_zero_copy)
            {
                ggml_backend_buffer_t hidden_out_buffer = nullptr;
                if (create_binding_from_host_ptr_2d(ctx, g_backend, hidden_out_desc, hidden_out_bind, hidden_out_buffer))
                {
                    new_entry->hidden_out_host_buffer = BufferHandle(hidden_out_buffer);
                    new_entry->hidden_out_zero_copy = true;
                }
            }
            if (hidden_out_bind.storage == nullptr)
            {
                hidden_out_bind = create_standard_binding(ctx, hidden_out_desc);
            }

            ggml_tensor* conv_state_t = conv_dim > 0
                ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, xbc_size, conv_dim)
                : nullptr;
            ggml_tensor* ssm_state_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, head_dim, n_head, 1);
            ggml_tensor* conv_weight_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, xbc_size);
            ggml_tensor* conv_bias_t = has_conv_bias ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, xbc_size) : nullptr;
            ggml_tensor* dt_bias_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_head);
            ggml_tensor* a_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_head);
            ggml_tensor* d_t = has_d ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_head) : nullptr;
            ggml_tensor* norm_t = has_norm ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inner_per_group, n_group) : nullptr;
            ggml_tensor* ids_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

            ggml_tensor* projected = projected_bind.tensor; // (d_in_proj_total, T)

            ggml_tensor* z = ggml_view_2d(
                ctx, projected,
                d_inner, T,
                projected->nb[1],
                0);
            z = ggml_cont(ctx, z);

            ggml_tensor* xbc = ggml_view_2d(
                ctx, projected,
                xbc_size, T,
                projected->nb[1],
                static_cast<std::size_t>(d_inner) * sizeof(float));
            ggml_tensor* xbc_time_major = ggml_cont(ctx, ggml_transpose(ctx, xbc)); // (T, xBC)

            ggml_tensor* conv_input = xbc_time_major;
            if (conv_dim > 0)
            {
                ggml_tensor* conv_state_time_major = ggml_cont(ctx, ggml_transpose(ctx, conv_state_t)); // (d_conv - 1, xBC)
                conv_input = ggml_concat(ctx, conv_state_time_major, xbc_time_major, 0);
            }
            conv_input = ggml_reshape_3d(ctx, conv_input, conv_dim + T, xbc_size, 1);

            ggml_tensor* conv = ggml_ssm_conv(ctx, conv_input, conv_weight_t); // (xBC, T, 1)
            if (has_conv_bias)
                conv = ggml_add(ctx, conv, conv_bias_t);
            conv = ggml_silu(ctx, conv);

            ggml_tensor* x_flat = ggml_view_2d(
                ctx, conv,
                d_inner, T,
                conv->nb[1],
                0);
            x_flat = ggml_cont(ctx, x_flat);
            ggml_tensor* x = ggml_reshape_4d(ctx, x_flat, head_dim, n_head, T, 1);

            const std::size_t b_offset = static_cast<std::size_t>(d_inner) * sizeof(float);
            const std::size_t c_offset = static_cast<std::size_t>(d_inner + n_group * d_state) * sizeof(float);
            ggml_tensor* b_flat = ggml_view_2d(
                ctx, conv,
                n_group * d_state, T,
                conv->nb[1],
                b_offset);
            ggml_tensor* c_flat = ggml_view_2d(
                ctx, conv,
                n_group * d_state, T,
                conv->nb[1],
                c_offset);
            b_flat = ggml_cont(ctx, b_flat);
            c_flat = ggml_cont(ctx, c_flat);
            ggml_tensor* b = ggml_reshape_4d(ctx, b_flat, d_state, n_group, T, 1);
            ggml_tensor* c = ggml_reshape_4d(ctx, c_flat, d_state, n_group, T, 1);

            const int dt_offset = 2 * d_inner + 2 * n_group * d_state;
            ggml_tensor* dt_raw = ggml_view_2d(
                ctx, projected,
                n_head, T,
                projected->nb[1],
                static_cast<std::size_t>(dt_offset) * sizeof(float));
            dt_raw = ggml_cont(ctx, dt_raw);
            ggml_tensor* dt = ggml_add(ctx, dt_raw, dt_bias_t);
            dt = ggml_reshape_3d(ctx, dt, n_head, T, 1);

            ggml_tensor* scan = ggml_ssm_scan(ctx, ssm_state_t, x, dt, a_t, b, c, ids_t);

            ggml_tensor* y = ggml_view_4d(
                ctx, scan,
                head_dim, n_head, T, 1,
                static_cast<std::size_t>(head_dim) * sizeof(float),
                static_cast<std::size_t>(d_inner) * sizeof(float),
                static_cast<std::size_t>(d_inner) * T * sizeof(float),
                0);

            if (has_d)
            {
                ggml_tensor* d4 = ggml_reshape_4d(ctx, d_t, 1, n_head, 1, 1);
                y = ggml_add(ctx, y, ggml_mul(ctx, x, d4));
            }

            ggml_tensor* y_flat = ggml_reshape_2d(ctx, y, d_inner, T);
            ggml_tensor* gated = ggml_mul(ctx, y_flat, ggml_silu(ctx, z));

            ggml_tensor* final_hidden = gated;
            if (has_norm)
            {
                ggml_tensor* grouped = ggml_reshape_3d(ctx, gated, inner_per_group, n_group, T);
                ggml_tensor* normed = ggml_rms_norm(ctx, grouped, eps);
                normed = ggml_mul(ctx, normed, norm_t);
                final_hidden = ggml_reshape_2d(ctx, normed, d_inner, T);
            }

            ggml_tensor* out_cpy = ggml_cpy(ctx, final_hidden, hidden_out_bind.tensor);

            const std::size_t y_bytes = static_cast<std::size_t>(d_inner) * T * sizeof(float);
            ggml_tensor* state_out = ggml_view_4d(
                ctx, scan,
                d_state, head_dim, n_head, 1,
                static_cast<std::size_t>(d_state) * sizeof(float),
                static_cast<std::size_t>(d_state) * head_dim * sizeof(float),
                static_cast<std::size_t>(d_state) * head_dim * n_head * sizeof(float),
                y_bytes);
            ggml_tensor* state_cpy = ggml_cpy(ctx, state_out, ssm_state_t);

            ggml_set_output(out_cpy);
            ggml_set_output(state_cpy);

            new_entry->graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);
            ggml_build_forward_expand(new_entry->graph, out_cpy);
            ggml_build_forward_expand(new_entry->graph, state_cpy);

            BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
            if (!buffer.value)
            {
                set_last_error("NemotronMamba2Prefill: buffer alloc failed.");
                return 0;
            }
            new_entry->buffer = std::move(buffer);

            new_entry->projected_storage = projected_bind.storage;
            new_entry->hidden_out_storage = hidden_out_bind.storage;
            new_entry->conv_state_storage = conv_state_t;
            new_entry->ssm_state_storage = ssm_state_t;
            new_entry->conv_weight_storage = conv_weight_t;
            new_entry->conv_bias_storage = conv_bias_t;
            new_entry->dt_bias_storage = dt_bias_t;
            new_entry->a_storage = a_t;
            new_entry->d_storage = d_t;
            new_entry->norm_storage = norm_t;
            new_entry->ids_storage = ids_t;

            new_entry->projected_bytes = projected_bind.raw_bytes;
            new_entry->hidden_out_bytes = hidden_out_bind.raw_bytes;
            new_entry->conv_state_bytes = static_cast<std::size_t>(conv_state_elements) * sizeof(float);
            new_entry->ssm_state_bytes = static_cast<std::size_t>(ssm_state_elements) * sizeof(float);
            new_entry->conv_weight_bytes = static_cast<std::size_t>(d_conv) * xbc_size * sizeof(float);
            new_entry->conv_bias_bytes = has_conv_bias ? static_cast<std::size_t>(xbc_size) * sizeof(float) : 0;
            new_entry->dt_bias_bytes = static_cast<std::size_t>(n_head) * sizeof(float);
            new_entry->a_bytes = static_cast<std::size_t>(n_head) * sizeof(float);
            new_entry->d_bytes = has_d ? static_cast<std::size_t>(n_head) * sizeof(float) : 0;
            new_entry->norm_bytes = has_norm ? static_cast<std::size_t>(d_inner) * sizeof(float) : 0;

            const std::int32_t zero_id = 0;
            ggml_backend_tensor_set(new_entry->ids_storage, &zero_id, 0, sizeof(zero_id));

            std::lock_guard<std::mutex> lk(g_mamba2_prefill_cache_mutex);
            auto [it, inserted] = g_mamba2_prefill_cache.emplace(cache_key, std::move(new_entry));
            entry = it->second.get();
            ensure_mamba2_prefill_cache_cleanup_registered();
        }

        {
            std::lock_guard<std::mutex> entry_lk(entry->compute_mutex);

            // The projection is produced by the preceding GGML matmul. When we can
            // bind it zero-copy, leave it on the Metal queue and let this graph
            // consume it in-order. Fall back to a barrier only when a CPU memcpy is
            // required to upload that projection into the cached graph buffer.
            if (!entry->projected_zero_copy)
            {
                host_read_barrier();
                ggml_backend_tensor_set(entry->projected_storage, projected_desc.data, 0, entry->projected_bytes);
            }

            if (conv_dim > 0)
                ggml_backend_tensor_set(entry->conv_state_storage, conv_state_data, 0, entry->conv_state_bytes);
            ggml_backend_tensor_set(entry->ssm_state_storage, ssm_state_data, 0, entry->ssm_state_bytes);
            ggml_backend_tensor_set(entry->conv_weight_storage, conv_weight_data, 0, entry->conv_weight_bytes);
            if (has_conv_bias)
                ggml_backend_tensor_set(entry->conv_bias_storage, conv_bias_data, 0, entry->conv_bias_bytes);
            ggml_backend_tensor_set(entry->dt_bias_storage, dt_bias_data, 0, entry->dt_bias_bytes);
            ggml_backend_tensor_set(entry->a_storage, a_data, 0, entry->a_bytes);
            if (has_d)
                ggml_backend_tensor_set(entry->d_storage, d_data, 0, entry->d_bytes);
            if (has_norm)
                ggml_backend_tensor_set(entry->norm_storage, ssm_norm_data, 0, entry->norm_bytes);

            ggml_status status = ggml_backend_graph_compute(g_backend, entry->graph);
            if (status != GGML_STATUS_SUCCESS)
            {
                set_last_error("NemotronMamba2Prefill: graph compute failed.");
                return 0;
            }
            ggml_backend_synchronize(g_backend);

            if (!entry->hidden_out_zero_copy)
                ggml_backend_tensor_get(entry->hidden_out_storage, hidden_out_desc.data, 0, entry->hidden_out_bytes);
            ggml_backend_tensor_get(entry->ssm_state_storage, ssm_state_data, 0, entry->ssm_state_bytes);

            update_conv_state_from_projected(
                static_cast<float*>(conv_state_data),
                projected_desc,
                d_inner,
                xbc_size,
                conv_dim);
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
        set_last_error("Unknown error in NemotronMamba2Prefill.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_NemotronMamba2DecodeF32(
    std::uint64_t state_key,
    TensorView2DDesc projected_desc,
    TensorView2DDesc hidden_out_desc,
    void* conv_state_data,
    int conv_state_elements,
    void* ssm_state_data,
    int ssm_state_elements,
    int initialize_state,
    int download_state,
    void* conv_weight_data,
    void* conv_bias_data,
    void* dt_bias_data,
    void* a_data,
    void* d_data,
    void* ssm_norm_data,
    int d_inner,
    int d_state,
    int n_head,
    int head_dim,
    int n_group,
    int d_conv,
    float eps)
{
    try
    {
        if (state_key == 0)
        {
            set_last_error("NemotronMamba2Decode: state key must be non-zero.");
            return 0;
        }

        if (!validate_mamba2_prefill_args(
                projected_desc, hidden_out_desc,
                conv_state_data, conv_state_elements,
                ssm_state_data, ssm_state_elements,
                conv_weight_data, dt_bias_data, a_data,
                d_inner, d_state, n_head, head_dim, n_group, d_conv))
        {
            return 0;
        }
        if (hidden_out_desc.dim0 != 1)
        {
            set_last_error("NemotronMamba2Decode: decode requires exactly one token.");
            return 0;
        }

        const int T = 1;
        const int conv_dim = d_conv - 1;
        const int xbc_size = d_inner + 2 * n_group * d_state;
        const int d_in_proj_total = 2 * d_inner + 2 * n_group * d_state + n_head;
        const int inner_per_group = d_inner / n_group;
        const bool has_conv_bias = conv_bias_data != nullptr;
        const bool has_d = d_data != nullptr;
        const bool has_norm = ssm_norm_data != nullptr;
        const std::size_t projected_bytes = static_cast<std::size_t>(projected_desc.raw_bytes);
        const std::size_t hidden_out_bytes = static_cast<std::size_t>(hidden_out_desc.raw_bytes);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        const bool projected_can_zero_copy =
            g_backend_type == BACKEND_TYPE_METAL &&
            projected_bytes >= 4096 &&
            can_use_host_ptr_buffer(g_backend, dev, projected_desc.data, projected_bytes);
        const bool hidden_out_can_zero_copy =
            g_backend_type == BACKEND_TYPE_METAL &&
            hidden_out_bytes >= 4096 &&
            can_use_host_ptr_buffer(g_backend, dev, hidden_out_desc.data, hidden_out_bytes);

        Mamba2DecodeCacheKey cache_key{
            state_key, d_in_proj_total, d_inner, d_state, n_head, head_dim, n_group, d_conv,
            has_conv_bias, has_d, has_norm,
            projected_can_zero_copy ? reinterpret_cast<std::uintptr_t>(projected_desc.data) : 0u,
            hidden_out_can_zero_copy ? reinterpret_cast<std::uintptr_t>(hidden_out_desc.data) : 0u,
            eps
        };

        Mamba2DecodeCacheEntry* entry = nullptr;
        {
            std::lock_guard<std::mutex> lk(g_mamba2_decode_cache_mutex);
            auto it = g_mamba2_decode_cache.find(cache_key);
            if (it != g_mamba2_decode_cache.end())
                entry = it->second.get();
        }

        if (entry == nullptr)
        {
            const std::size_t per_tensor_bytes = 384;
            std::size_t ctx_size = 160 * per_tensor_bytes;
            if (ctx_size < 4 * 1024 * 1024) ctx_size = 4 * 1024 * 1024;

            auto new_entry = std::make_unique<Mamba2DecodeCacheEntry>();
            new_entry->ctx_buffer = std::make_unique<std::uint8_t[]>(ctx_size);

            ggml_init_params params = {};
            params.mem_size = ctx_size;
            params.mem_buffer = new_entry->ctx_buffer.get();
            params.no_alloc = true;
            new_entry->ctx = ggml_init(params);
            if (new_entry->ctx == nullptr)
            {
                set_last_error("NemotronMamba2Decode: cached context init failed.");
                return 0;
            }

            ggml_context* ctx = new_entry->ctx;

            TensorBinding projected_bind{};
            TensorBinding hidden_out_bind{};

            if (projected_can_zero_copy)
            {
                ggml_backend_buffer_t projected_buffer = nullptr;
                if (create_binding_from_host_ptr_2d(ctx, g_backend, projected_desc, projected_bind, projected_buffer))
                {
                    new_entry->projected_host_buffer = BufferHandle(projected_buffer);
                    new_entry->projected_zero_copy = true;
                }
            }
            if (projected_bind.storage == nullptr)
                projected_bind = create_standard_binding(ctx, projected_desc);

            if (hidden_out_can_zero_copy)
            {
                ggml_backend_buffer_t hidden_out_buffer = nullptr;
                if (create_binding_from_host_ptr_2d(ctx, g_backend, hidden_out_desc, hidden_out_bind, hidden_out_buffer))
                {
                    new_entry->hidden_out_host_buffer = BufferHandle(hidden_out_buffer);
                    new_entry->hidden_out_zero_copy = true;
                }
            }
            if (hidden_out_bind.storage == nullptr)
                hidden_out_bind = create_standard_binding(ctx, hidden_out_desc);

            ggml_tensor* conv_state_t = conv_dim > 0
                ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, xbc_size, conv_dim)
                : nullptr;
            ggml_tensor* ssm_state_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, head_dim, n_head, 1);
            ggml_tensor* conv_weight_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, xbc_size);
            ggml_tensor* conv_bias_t = has_conv_bias ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, xbc_size) : nullptr;
            ggml_tensor* dt_bias_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_head);
            ggml_tensor* a_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_head);
            ggml_tensor* d_t = has_d ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_head) : nullptr;
            ggml_tensor* norm_t = has_norm ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inner_per_group, n_group) : nullptr;
            ggml_tensor* ids_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

            ggml_tensor* projected = projected_bind.tensor; // (d_in_proj_total, 1)

            ggml_tensor* z = ggml_view_2d(
                ctx, projected,
                d_inner, T,
                projected->nb[1],
                0);
            z = ggml_cont(ctx, z);

            ggml_tensor* xbc = ggml_view_2d(
                ctx, projected,
                xbc_size, T,
                projected->nb[1],
                static_cast<std::size_t>(d_inner) * sizeof(float));
            ggml_tensor* xbc_time_major = ggml_cont(ctx, ggml_transpose(ctx, xbc)); // (1, xBC)

            ggml_tensor* conv_input = xbc_time_major;
            if (conv_dim > 0)
            {
                ggml_tensor* conv_state_time_major = ggml_cont(ctx, ggml_transpose(ctx, conv_state_t));
                conv_input = ggml_concat(ctx, conv_state_time_major, xbc_time_major, 0);
            }
            conv_input = ggml_reshape_3d(ctx, conv_input, conv_dim + T, xbc_size, 1);

            ggml_tensor* conv = ggml_ssm_conv(ctx, conv_input, conv_weight_t); // (xBC, 1, 1)
            if (has_conv_bias)
                conv = ggml_add(ctx, conv, conv_bias_t);
            conv = ggml_silu(ctx, conv);

            ggml_tensor* x_flat = ggml_view_2d(
                ctx, conv,
                d_inner, T,
                conv->nb[1],
                0);
            x_flat = ggml_cont(ctx, x_flat);
            ggml_tensor* x = ggml_reshape_4d(ctx, x_flat, head_dim, n_head, T, 1);

            const std::size_t b_offset = static_cast<std::size_t>(d_inner) * sizeof(float);
            const std::size_t c_offset = static_cast<std::size_t>(d_inner + n_group * d_state) * sizeof(float);
            ggml_tensor* b_flat = ggml_view_2d(
                ctx, conv,
                n_group * d_state, T,
                conv->nb[1],
                b_offset);
            ggml_tensor* c_flat = ggml_view_2d(
                ctx, conv,
                n_group * d_state, T,
                conv->nb[1],
                c_offset);
            b_flat = ggml_cont(ctx, b_flat);
            c_flat = ggml_cont(ctx, c_flat);
            ggml_tensor* b = ggml_reshape_4d(ctx, b_flat, d_state, n_group, T, 1);
            ggml_tensor* c = ggml_reshape_4d(ctx, c_flat, d_state, n_group, T, 1);

            const int dt_offset = 2 * d_inner + 2 * n_group * d_state;
            ggml_tensor* dt_raw = ggml_view_2d(
                ctx, projected,
                n_head, T,
                projected->nb[1],
                static_cast<std::size_t>(dt_offset) * sizeof(float));
            dt_raw = ggml_cont(ctx, dt_raw);
            ggml_tensor* dt = ggml_add(ctx, dt_raw, dt_bias_t);
            dt = ggml_reshape_3d(ctx, dt, n_head, T, 1);

            ggml_tensor* scan = ggml_ssm_scan(ctx, ssm_state_t, x, dt, a_t, b, c, ids_t);

            ggml_tensor* y = ggml_view_4d(
                ctx, scan,
                head_dim, n_head, T, 1,
                static_cast<std::size_t>(head_dim) * sizeof(float),
                static_cast<std::size_t>(d_inner) * sizeof(float),
                static_cast<std::size_t>(d_inner) * T * sizeof(float),
                0);

            if (has_d)
            {
                ggml_tensor* d4 = ggml_reshape_4d(ctx, d_t, 1, n_head, 1, 1);
                y = ggml_add(ctx, y, ggml_mul(ctx, x, d4));
            }

            ggml_tensor* y_flat = ggml_reshape_2d(ctx, y, d_inner, T);
            ggml_tensor* gated = ggml_mul(ctx, y_flat, ggml_silu(ctx, z));

            ggml_tensor* final_hidden = gated;
            if (has_norm)
            {
                ggml_tensor* grouped = ggml_reshape_3d(ctx, gated, inner_per_group, n_group, T);
                ggml_tensor* normed = ggml_rms_norm(ctx, grouped, eps);
                normed = ggml_mul(ctx, normed, norm_t);
                final_hidden = ggml_reshape_2d(ctx, normed, d_inner, T);
            }

            ggml_tensor* out_cpy = ggml_cpy(ctx, final_hidden, hidden_out_bind.tensor);

            ggml_tensor* conv_state_cpy = nullptr;
            if (conv_dim > 0)
            {
                ggml_tensor* xbc_state_row = ggml_reshape_2d(ctx, xbc, xbc_size, 1);
                ggml_tensor* next_conv_state = xbc_state_row;
                if (conv_dim > 1)
                {
                    ggml_tensor* conv_tail = ggml_view_2d(
                        ctx, conv_state_t,
                        xbc_size, conv_dim - 1,
                        conv_state_t->nb[1],
                        static_cast<std::size_t>(xbc_size) * sizeof(float));
                    next_conv_state = ggml_concat(ctx, conv_tail, xbc_state_row, 1);
                }
                // Force the conv-state write after the conv/scan branch has read
                // the old state. Without this dependency, the state rotation is
                // otherwise independent of the main output graph.
                ggml_tensor* conv_dep = ggml_sum(ctx, final_hidden);
                conv_dep = ggml_scale(ctx, conv_dep, 0.0f);
                next_conv_state = ggml_add(ctx, next_conv_state, conv_dep);
                conv_state_cpy = ggml_cpy(ctx, next_conv_state, conv_state_t);
            }

            const std::size_t y_bytes = static_cast<std::size_t>(d_inner) * T * sizeof(float);
            ggml_tensor* state_out = ggml_view_4d(
                ctx, scan,
                d_state, head_dim, n_head, 1,
                static_cast<std::size_t>(d_state) * sizeof(float),
                static_cast<std::size_t>(d_state) * head_dim * sizeof(float),
                static_cast<std::size_t>(d_state) * head_dim * n_head * sizeof(float),
                y_bytes);
            ggml_tensor* state_cpy = ggml_cpy(ctx, state_out, ssm_state_t);

            ggml_set_output(out_cpy);
            if (conv_state_cpy != nullptr)
                ggml_set_output(conv_state_cpy);
            ggml_set_output(state_cpy);

            new_entry->graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);
            ggml_build_forward_expand(new_entry->graph, out_cpy);
            if (conv_state_cpy != nullptr)
                ggml_build_forward_expand(new_entry->graph, conv_state_cpy);
            ggml_build_forward_expand(new_entry->graph, state_cpy);

            BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
            if (!buffer.value)
            {
                set_last_error("NemotronMamba2Decode: buffer alloc failed.");
                return 0;
            }
            new_entry->buffer = std::move(buffer);

            new_entry->projected_storage = projected_bind.storage;
            new_entry->hidden_out_storage = hidden_out_bind.storage;
            new_entry->conv_state_storage = conv_state_t;
            new_entry->ssm_state_storage = ssm_state_t;
            new_entry->conv_weight_storage = conv_weight_t;
            new_entry->conv_bias_storage = conv_bias_t;
            new_entry->dt_bias_storage = dt_bias_t;
            new_entry->a_storage = a_t;
            new_entry->d_storage = d_t;
            new_entry->norm_storage = norm_t;
            new_entry->ids_storage = ids_t;

            new_entry->projected_bytes = projected_bind.raw_bytes;
            new_entry->hidden_out_bytes = hidden_out_bind.raw_bytes;
            new_entry->conv_state_bytes = static_cast<std::size_t>(conv_state_elements) * sizeof(float);
            new_entry->ssm_state_bytes = static_cast<std::size_t>(ssm_state_elements) * sizeof(float);
            new_entry->conv_weight_bytes = static_cast<std::size_t>(d_conv) * xbc_size * sizeof(float);
            new_entry->conv_bias_bytes = has_conv_bias ? static_cast<std::size_t>(xbc_size) * sizeof(float) : 0;
            new_entry->dt_bias_bytes = static_cast<std::size_t>(n_head) * sizeof(float);
            new_entry->a_bytes = static_cast<std::size_t>(n_head) * sizeof(float);
            new_entry->d_bytes = has_d ? static_cast<std::size_t>(n_head) * sizeof(float) : 0;
            new_entry->norm_bytes = has_norm ? static_cast<std::size_t>(d_inner) * sizeof(float) : 0;

            const std::int32_t zero_id = 0;
            ggml_backend_tensor_set(new_entry->ids_storage, &zero_id, 0, sizeof(zero_id));

            std::lock_guard<std::mutex> lk(g_mamba2_decode_cache_mutex);
            auto [it, inserted] = g_mamba2_decode_cache.emplace(cache_key, std::move(new_entry));
            entry = it->second.get();
            ensure_mamba2_prefill_cache_cleanup_registered();
        }

        {
            std::lock_guard<std::mutex> entry_lk(entry->compute_mutex);

            if (!entry->projected_zero_copy)
            {
                host_read_barrier();
                ggml_backend_tensor_set(entry->projected_storage, projected_desc.data, 0, entry->projected_bytes);
            }

            if (!entry->weights_uploaded)
            {
                ggml_backend_tensor_set(entry->conv_weight_storage, conv_weight_data, 0, entry->conv_weight_bytes);
                if (has_conv_bias)
                    ggml_backend_tensor_set(entry->conv_bias_storage, conv_bias_data, 0, entry->conv_bias_bytes);
                ggml_backend_tensor_set(entry->dt_bias_storage, dt_bias_data, 0, entry->dt_bias_bytes);
                ggml_backend_tensor_set(entry->a_storage, a_data, 0, entry->a_bytes);
                if (has_d)
                    ggml_backend_tensor_set(entry->d_storage, d_data, 0, entry->d_bytes);
                if (has_norm)
                    ggml_backend_tensor_set(entry->norm_storage, ssm_norm_data, 0, entry->norm_bytes);
                entry->weights_uploaded = true;
            }

            if (initialize_state || !entry->state_initialized)
            {
                if (conv_dim > 0)
                    ggml_backend_tensor_set(entry->conv_state_storage, conv_state_data, 0, entry->conv_state_bytes);
                ggml_backend_tensor_set(entry->ssm_state_storage, ssm_state_data, 0, entry->ssm_state_bytes);
                entry->state_initialized = true;
            }

            ggml_status status = ggml_backend_graph_compute(g_backend, entry->graph);
            if (status != GGML_STATUS_SUCCESS)
            {
                set_last_error("NemotronMamba2Decode: graph compute failed.");
                return 0;
            }

            if (download_state)
            {
                ggml_backend_synchronize(g_backend);
                if (!entry->hidden_out_zero_copy)
                    ggml_backend_tensor_get(entry->hidden_out_storage, hidden_out_desc.data, 0, entry->hidden_out_bytes);
                if (conv_dim > 0)
                    ggml_backend_tensor_get(entry->conv_state_storage, conv_state_data, 0, entry->conv_state_bytes);
                ggml_backend_tensor_get(entry->ssm_state_storage, ssm_state_data, 0, entry->ssm_state_bytes);
            }
            else
            {
                finalize_compute(entry->hidden_out_zero_copy, entry->hidden_out_storage, hidden_out_desc.data, entry->hidden_out_bytes);
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
        set_last_error("Unknown error in NemotronMamba2Decode.");
        return 0;
    }
}

TSG_EXPORT void TSGgml_NemotronMamba2DecodeClear(std::uint64_t model_key)
{
    host_read_barrier();
    std::lock_guard<std::mutex> lk(g_mamba2_decode_cache_mutex);
    if (model_key == 0)
    {
        g_mamba2_decode_cache.clear();
        return;
    }

    for (auto it = g_mamba2_decode_cache.begin(); it != g_mamba2_decode_cache.end();)
    {
        if ((it->first.state_key >> 32) == model_key)
            it = g_mamba2_decode_cache.erase(it);
        else
            ++it;
    }
}

// ============================================================================
// TSGgml_NemotronMamba2BatchedStepF32
// ============================================================================
// vLLM-style batched per-token Mamba2 compute. Replaces N separate calls to
// the per-seq Mamba2Block path with one native dispatch that walks every
// (seq, token) pair, indexing into per-sequence conv + SSM state via the
// descriptors. Mirrors my Qwen 3.5 Phase 7 GDN batched kernel structure.
//
// Math is a faithful port of the C# Mamba2SSMStepSIMD + Mamba2Conv1dStep + the
// post-process (SwiGLU + group RMSNorm) in NemotronModel.cs's Mamba2Forward
// fallback path. Heads are independent within a single token so we dispatch
// across them via GCD on macOS, matching the C# Parallel.For pattern.
//
// Decode-only (single token per seq is the common case for serving). Prefill
// can fall through to the existing single-seq native kernel which already
// handles chunked GGML graphs.
#if defined(__ARM_NEON)
// NEON intrinsics already included for the GDN kernel
#endif

#if defined(__APPLE__)
// dispatch already included for the GDN kernel
#endif

namespace
{
    struct NemoMamba2BatchedSeqDesc
    {
        int32_t seq_start;       // start row in batched packed input
        int32_t seq_len;         // tokens for this seq
        int32_t pad0;
        int32_t pad1;
        void*   conv_state;      // float[(d_conv-1) * xbc_size]
        void*   ssm_state;       // float[n_head * head_dim * d_state]
    };

    // Softplus / sigmoid / silu helpers — same as in the GDN kernel.
    static inline float softplus_f(float x)
    {
        if (x > 20.0f) return x;
        if (x < -20.0f) return std::exp(x);
        return std::log1p(std::exp(x));
    }
    static inline float sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }
    static inline float silu_f(float x) { return x * sigmoid_f(x); }

#if defined(__ARM_NEON)
    static inline float nemo_vec_dot(const float* a, const float* b, int n)
    {
        int i = 0;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        for (; i <= n - 4; i += 4)
            acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        float s = vaddvq_f32(acc0);
        for (; i < n; i++) s += a[i] * b[i];
        return s;
    }
    static inline float nemo_vec_sumsq(const float* a, int n)
    {
        int i = 0;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        for (; i <= n - 4; i += 4)
        {
            float32x4_t v = vld1q_f32(a + i);
            acc0 = vfmaq_f32(acc0, v, v);
        }
        float s = vaddvq_f32(acc0);
        for (; i < n; i++) s += a[i] * a[i];
        return s;
    }
#else
    static inline float nemo_vec_dot(const float* a, const float* b, int n)
    { float s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }
    static inline float nemo_vec_sumsq(const float* a, int n)
    { float s = 0; for (int i = 0; i < n; i++) s += a[i] * a[i]; return s; }
#endif
}

TSG_EXPORT int TSGgml_NemotronMamba2BatchedStepF32(
    int32_t                    num_seqs,
    NemoMamba2BatchedSeqDesc*  seqs,
    int32_t                    num_tokens,
    const float*               packed_batched,   // [num_tokens, d_in_proj_total]
    int32_t                    d_in_proj_total,
    int32_t                    d_inner,
    int32_t                    d_state,
    int32_t                    n_head,
    int32_t                    head_dim,
    int32_t                    n_group,
    int32_t                    d_conv,
    const float*               conv_wt,          // [d_conv, xbc_size] transposed (matches _mamba2ConvWT layout)
    const float*               conv_bias,        // [xbc_size] nullable
    const float*               dt_bias,          // [n_head]
    const float*               a_log,            // [n_head] (literal "A" in the math, not log — matches _ssmAW naming)
    const float*               d_data,           // [n_head] nullable
    const float*               ssm_norm_w,       // [d_inner] nullable
    float                      eps,
    float*                     out_batched)      // [num_tokens, d_inner]
{
    try
    {
        if (num_seqs <= 0 || seqs == nullptr ||
            packed_batched == nullptr || out_batched == nullptr ||
            conv_wt == nullptr || dt_bias == nullptr || a_log == nullptr)
        {
            set_last_error("Mamba2BatchedStep: null arg or num_seqs<=0.");
            return 0;
        }

        const int xbc_size = d_inner + 2 * n_group * d_state;
        const int conv_dim = d_conv - 1;
        const int heads_per_group = n_head / n_group;
        const int state_per_head = head_dim * d_state;
        const int inner_per_group = d_inner / n_group;

        // Per-call scratch: conv_out + y. Heads share state per-token but each
        // (h, d) is independent during the SSM scan inner loop.
        std::vector<float> conv_out(xbc_size);
        std::vector<float> y_buf(d_inner);

        for (int si = 0; si < num_seqs; si++)
        {
            NemoMamba2BatchedSeqDesc& sd = seqs[si];
            float* conv_state = static_cast<float*>(sd.conv_state);
            float* ssm_state  = static_cast<float*>(sd.ssm_state);
            const int seq_start = sd.seq_start;
            const int seq_len   = sd.seq_len;
            if (seq_len <= 0) continue;
            if (conv_state == nullptr || ssm_state == nullptr)
            {
                set_last_error("Mamba2BatchedStep: per-seq state pointer is null.");
                return 0;
            }

            for (int t = 0; t < seq_len; t++)
            {
                const float* row = packed_batched + static_cast<long>(seq_start + t) * d_in_proj_total;
                const float* z_ptr  = row;
                const float* xbc_in = row + d_inner;
                const float* dt_raw = row + 2 * d_inner + 2 * n_group * d_state;

                // --- Conv1d step ---
                // conv_out[ch] = sum_{ki<convDim} state[ki][ch]*conv_wt[ki][ch]
                //              + xbc_in[ch] * conv_wt[convDim][ch]
                //              + conv_bias[ch]
                if (conv_dim > 0)
                {
                    // tap 0
                    const float* wt0 = conv_wt;
                    for (int ch = 0; ch < xbc_size; ch++)
                        conv_out[ch] = conv_state[ch] * wt0[ch];
                    for (int ki = 1; ki < conv_dim; ki++)
                    {
                        const float* sp = conv_state + static_cast<long>(ki) * xbc_size;
                        const float* wp = conv_wt + static_cast<long>(ki) * xbc_size;
                        for (int ch = 0; ch < xbc_size; ch++)
                            conv_out[ch] += sp[ch] * wp[ch];
                    }
                    // current input tap
                    const float* wpN = conv_wt + static_cast<long>(conv_dim) * xbc_size;
                    for (int ch = 0; ch < xbc_size; ch++)
                        conv_out[ch] += xbc_in[ch] * wpN[ch];
                }
                else
                {
                    for (int ch = 0; ch < xbc_size; ch++)
                        conv_out[ch] = xbc_in[ch] * conv_wt[ch];
                }
                if (conv_bias != nullptr)
                {
                    for (int ch = 0; ch < xbc_size; ch++)
                        conv_out[ch] += conv_bias[ch];
                }
                // SiLU in place on conv_out.
                for (int ch = 0; ch < xbc_size; ch++) conv_out[ch] = silu_f(conv_out[ch]);

                // Shift conv state by one (drop oldest, append xbc_in).
                if (conv_dim > 1)
                {
                    // memmove rows [1..conv_dim-1] down to [0..conv_dim-2]
                    std::memmove(conv_state,
                                 conv_state + xbc_size,
                                 static_cast<std::size_t>(conv_dim - 1) * xbc_size * sizeof(float));
                }
                if (conv_dim > 0)
                {
                    std::memcpy(conv_state + static_cast<long>(conv_dim - 1) * xbc_size,
                                xbc_in,
                                static_cast<std::size_t>(xbc_size) * sizeof(float));
                }

                // Split conv_out into x (first d_inner) | B (n_group*d_state) | C (n_group*d_state).
                const float* x_base = conv_out.data();
                const float* b_base = x_base + d_inner;
                const float* c_base = b_base + n_group * d_state;

                // --- SSM scan step (heads in parallel via GCD on macOS) ---
                float* y_ptr = y_buf.data();
                auto head_body = [&](int h)
                {
                    float dt_softplus = softplus_f(dt_raw[h] + dt_bias[h]);
                    float dA = std::exp(dt_softplus * a_log[h]);
                    int g = h / heads_per_group;
                    float* state_h = ssm_state + static_cast<long>(h) * state_per_head;
                    const float* x_h = x_base + h * head_dim;
                    float* y_h = y_ptr + h * head_dim;
                    const float* b_g = b_base + g * d_state;
                    const float* c_g = c_base + g * d_state;

                    for (int d = 0; d < head_dim; d++)
                    {
                        float xDt = x_h[d] * dt_softplus;
                        float* state_col = state_h + d * d_state;
                        float sum = 0.0f;
                        for (int s2 = 0; s2 < d_state; s2++)
                        {
                            state_col[s2] = state_col[s2] * dA + b_g[s2] * xDt;
                            sum += state_col[s2] * c_g[s2];
                        }
                        y_h[d] = sum;
                    }
                    if (d_data != nullptr)
                    {
                        float d_h = d_data[h];
                        for (int d = 0; d < head_dim; d++) y_h[d] += d_h * x_h[d];
                    }
                };
#if defined(__APPLE__)
                if (n_head >= 4)
                {
                    dispatch_apply(static_cast<size_t>(n_head),
                                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                                   ^(size_t h) { head_body(static_cast<int>(h)); });
                }
                else
                {
                    for (int h = 0; h < n_head; h++) head_body(h);
                }
#else
                for (int h = 0; h < n_head; h++) head_body(h);
#endif

                // --- SwiGLU: y = silu(z) * y ---
                for (int i = 0; i < d_inner; i++) y_ptr[i] = silu_f(z_ptr[i]) * y_ptr[i];

                // --- Group RMSNorm ---
                if (ssm_norm_w != nullptr)
                {
                    for (int g = 0; g < n_group; g++)
                    {
                        int offset = g * inner_per_group;
                        float ss = nemo_vec_sumsq(y_ptr + offset, inner_per_group);
                        float rms_inv = 1.0f / std::sqrt(ss / inner_per_group + eps);
                        for (int i = 0; i < inner_per_group; i++)
                            y_ptr[offset + i] = y_ptr[offset + i] * rms_inv * ssm_norm_w[offset + i];
                    }
                }

                // Write to batched output.
                float* out_row = out_batched + static_cast<long>(seq_start + t) * d_inner;
                std::memcpy(out_row, y_ptr, static_cast<std::size_t>(d_inner) * sizeof(float));
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
        set_last_error("Unknown error in Mamba2BatchedStep.");
        return 0;
    }
}

