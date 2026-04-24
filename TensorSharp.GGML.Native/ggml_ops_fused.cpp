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

using namespace tsg;

namespace {

// ============================================================================
// Fused RMSNorm + Quantized MatMul: single GPU dispatch for two ops.
// result = matmul(rms_norm(input, norm_weight, eps), quant_weight)
// ============================================================================

int fused_rms_norm_matmul_quant_f32_impl(
    const TensorView2DDesc& result_desc,
    const TensorView2DDesc& input_desc,
    void* norm_weight_data,
    int norm_weight_count,
    float eps,
    const QuantizedWeightDesc& m2_quant)
{
    if (!ensure_backend())
        return 0;

    if (!validate_desc(result_desc, "result") || !validate_desc(input_desc, "input"))
        return 0;

    if (norm_weight_data == nullptr || norm_weight_count <= 0)
    {
        set_last_error("Invalid norm weight data.");
        return 0;
    }

    if (m2_quant.data == nullptr || m2_quant.ne0 <= 0 || m2_quant.ne1 <= 0 || m2_quant.raw_bytes <= 0)
    {
        set_last_error("Invalid quantized weight descriptor for fused rms_norm_matmul.");
        return 0;
    }

    const int rows = input_desc.dim0;
    const int in_dim = input_desc.dim1;
    const int out_dim = result_desc.dim1;

    if (result_desc.dim0 != rows)
    {
        set_last_error("Size mismatch: result.dim0 != input.dim0 in fused rms_norm_matmul.");
        return 0;
    }

    const std::size_t ctx_size = 2 * 1024 * 1024;
    PooledContextHandle context;
    if (!context.init(ctx_size))
    {
        set_last_error("Failed to create ggml context for fused rms_norm_matmul.");
        return 0;
    }

    std::vector<BufferHandle> host_ptr_buffers;
    bool use_zero_copy = can_map_standard_view(input_desc);

    TensorBinding result_binding;
    TensorBinding input_binding;
    std::vector<float> packed_input;

    if (use_zero_copy)
    {
        ggml_backend_buffer_t result_buf = nullptr;
        ggml_backend_buffer_t input_buf = nullptr;
        const bool result_ok = create_binding_from_host_ptr_2d(context.value, g_backend, result_desc, result_binding, result_buf);
        const bool input_ok = result_ok && create_binding_from_host_ptr_2d(context.value, g_backend, input_desc, input_binding, input_buf);

        if (result_ok && input_ok)
        {
            host_ptr_buffers.emplace_back(result_buf);
            host_ptr_buffers.emplace_back(input_buf);
        }
        else
        {
            if (input_buf != nullptr) ggml_backend_buffer_free(input_buf);
            if (result_buf != nullptr) ggml_backend_buffer_free(result_buf);
            use_zero_copy = false;
            result_binding = create_standard_binding(context.value, result_desc);
            input_binding = can_map_standard_view(input_desc)
                ? create_standard_binding(context.value, input_desc)
                : create_packed_standard_binding(context.value, input_desc, packed_input);
        }
    }
    else
    {
        result_binding = create_standard_binding(context.value, result_desc);
        input_binding = can_map_standard_view(input_desc)
            ? create_standard_binding(context.value, input_desc)
            : create_packed_standard_binding(context.value, input_desc, packed_input);
    }

    // Norm weight tensor (1D float)
    ggml_tensor* norm_w_tensor = ggml_new_tensor_1d(context.value, GGML_TYPE_F32, norm_weight_count);

    // Quantized weight tensor
    ggml_type qtype = static_cast<ggml_type>(m2_quant.ggml_type);
    ggml_tensor* m2_tensor = ggml_new_tensor_2d(context.value, qtype, m2_quant.ne0, m2_quant.ne1);
    TensorBinding m2_binding = { m2_tensor, m2_tensor, static_cast<std::size_t>(m2_quant.raw_bytes) };

    if (result_binding.storage == nullptr || input_binding.storage == nullptr ||
        norm_w_tensor == nullptr || m2_tensor == nullptr)
    {
        set_last_error("Failed to allocate ggml tensors for fused rms_norm_matmul.");
        return 0;
    }

    // Cache quantized weight buffer
    bool m2_bound = false;
    bool m2_needs_upload = false;
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        if (dev != nullptr && m2_quant.raw_bytes >= 4096)
        {
            ggml_backend_buffer_t buf = nullptr;
            void* addr = nullptr;
            if (try_get_cacheable_tensor_buffer(g_backend, dev, m2_tensor,
                    m2_quant.data, static_cast<std::size_t>(m2_quant.raw_bytes),
                    buf, addr, m2_needs_upload))
            {
                ggml_status st = ggml_backend_tensor_alloc(buf, m2_tensor, addr);
                m2_bound = (st == GGML_STATUS_SUCCESS);
                if (!m2_bound) invalidate_cached_buffer(m2_quant.data);
            }
        }
    }

    // Cache norm weight buffer
    bool norm_bound = false;
    bool norm_needs_upload = false;
    std::size_t norm_bytes = static_cast<std::size_t>(norm_weight_count) * sizeof(float);
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        if (dev != nullptr && norm_bytes >= 4096)
        {
            ggml_backend_buffer_t buf = nullptr;
            void* addr = nullptr;
            if (try_get_cacheable_tensor_buffer(g_backend, dev, norm_w_tensor,
                    norm_weight_data, norm_bytes,
                    buf, addr, norm_needs_upload))
            {
                ggml_status st = ggml_backend_tensor_alloc(buf, norm_w_tensor, addr);
                norm_bound = (st == GGML_STATUS_SUCCESS);
                if (!norm_bound) invalidate_cached_buffer(norm_weight_data);
            }
        }
    }

    // Build graph: rms_norm → mul(gamma) → reshape_2d → mul_mat → cpy
    ggml_tensor* contiguous_input = ggml_cont(context.value, input_binding.tensor);
    ggml_tensor* normed = ggml_rms_norm(context.value, contiguous_input, eps);
    ggml_tensor* scaled = ggml_mul(context.value, normed, norm_w_tensor);

    ggml_tensor* scaled_2d = (rows == 1)
        ? ggml_reshape_2d(context.value, scaled, in_dim, 1)
        : scaled;

    ggml_tensor* mm = ggml_mul_mat(context.value, m2_binding.tensor, scaled_2d);
    ggml_tensor* output_tensor = ggml_cpy(context.value, mm, result_binding.tensor);
    ggml_set_output(output_tensor);

    ggml_cgraph* graph = ggml_new_graph(context.value);
    ggml_build_forward_expand(graph, output_tensor);

    BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
    if (buffer.value == nullptr)
    {
        set_last_error("Failed to allocate backend buffer for fused rms_norm_matmul.");
        return 0;
    }

    // Upload data
    if (!use_zero_copy)
    {
        if (packed_input.empty())
            upload_binding(input_binding, input_desc.data, input_binding.raw_bytes);
        else
            upload_binding(input_binding, packed_input.data(), input_binding.raw_bytes);
        if (result_binding.raw_bytes > logical_bytes(result_desc))
            upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
    }

    if (!norm_bound || norm_needs_upload)
        ggml_backend_tensor_set(norm_w_tensor, norm_weight_data, 0, norm_bytes);

    if (!m2_bound || m2_needs_upload)
        upload_binding(m2_binding, m2_quant.data, m2_binding.raw_bytes);

    ggml_status status = ggml_backend_graph_compute(g_backend, graph);
    if (status != GGML_STATUS_SUCCESS)
    {
        set_last_error("Graph execution failed for fused rms_norm_matmul.");
        return 0;
    }
    ggml_backend_synchronize(g_backend);

    if (!use_zero_copy)
        ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

    clear_last_error();
    return 1;
}

// ============================================================================
// Fused Quantized MatMul + Add: single GPU dispatch.
// residual += matmul(input, quant_weight)
// ============================================================================

int fused_matmul_quant_add_f32_impl(
    const TensorView2DDesc& residual_desc,
    const TensorView2DDesc& input_desc,
    const QuantizedWeightDesc& m2_quant)
{
    if (!ensure_backend())
        return 0;

    if (!validate_desc(residual_desc, "residual") || !validate_desc(input_desc, "input"))
        return 0;

    if (m2_quant.data == nullptr || m2_quant.ne0 <= 0 || m2_quant.ne1 <= 0 || m2_quant.raw_bytes <= 0)
    {
        set_last_error("Invalid quantized weight descriptor for fused matmul_add.");
        return 0;
    }

    const int rows = input_desc.dim0;
    const int in_dim = input_desc.dim1;
    const int out_dim = residual_desc.dim1;

    if (residual_desc.dim0 != rows)
    {
        set_last_error("Size mismatch: residual.dim0 != input.dim0 in fused matmul_add.");
        return 0;
    }

    const std::size_t ctx_size = 2 * 1024 * 1024;
    PooledContextHandle context;
    if (!context.init(ctx_size))
    {
        set_last_error("Failed to create ggml context for fused matmul_add.");
        return 0;
    }

    std::vector<BufferHandle> host_ptr_buffers;
    bool use_zero_copy = can_map_standard_view(input_desc);

    TensorBinding residual_binding;
    TensorBinding input_binding;
    std::vector<float> packed_input;

    if (use_zero_copy)
    {
        ggml_backend_buffer_t res_buf = nullptr;
        ggml_backend_buffer_t inp_buf = nullptr;
        const bool res_ok = create_binding_from_host_ptr_2d(context.value, g_backend, residual_desc, residual_binding, res_buf);
        const bool inp_ok = res_ok && create_binding_from_host_ptr_2d(context.value, g_backend, input_desc, input_binding, inp_buf);

        if (res_ok && inp_ok)
        {
            host_ptr_buffers.emplace_back(res_buf);
            host_ptr_buffers.emplace_back(inp_buf);
        }
        else
        {
            if (inp_buf != nullptr) ggml_backend_buffer_free(inp_buf);
            if (res_buf != nullptr) ggml_backend_buffer_free(res_buf);
            use_zero_copy = false;
            residual_binding = create_standard_binding(context.value, residual_desc);
            input_binding = can_map_standard_view(input_desc)
                ? create_standard_binding(context.value, input_desc)
                : create_packed_standard_binding(context.value, input_desc, packed_input);
        }
    }
    else
    {
        residual_binding = create_standard_binding(context.value, residual_desc);
        input_binding = can_map_standard_view(input_desc)
            ? create_standard_binding(context.value, input_desc)
            : create_packed_standard_binding(context.value, input_desc, packed_input);
    }

    // Quantized weight tensor
    ggml_type qtype = static_cast<ggml_type>(m2_quant.ggml_type);
    ggml_tensor* m2_tensor = ggml_new_tensor_2d(context.value, qtype, m2_quant.ne0, m2_quant.ne1);
    TensorBinding m2_binding = { m2_tensor, m2_tensor, static_cast<std::size_t>(m2_quant.raw_bytes) };

    if (residual_binding.storage == nullptr || input_binding.storage == nullptr || m2_tensor == nullptr)
    {
        set_last_error("Failed to allocate ggml tensors for fused matmul_add.");
        return 0;
    }

    // Cache quantized weight
    bool m2_bound = false;
    bool m2_needs_upload = false;
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        if (dev != nullptr && m2_quant.raw_bytes >= 4096)
        {
            ggml_backend_buffer_t buf = nullptr;
            void* addr = nullptr;
            if (try_get_cacheable_tensor_buffer(g_backend, dev, m2_tensor,
                    m2_quant.data, static_cast<std::size_t>(m2_quant.raw_bytes),
                    buf, addr, m2_needs_upload))
            {
                ggml_status st = ggml_backend_tensor_alloc(buf, m2_tensor, addr);
                m2_bound = (st == GGML_STATUS_SUCCESS);
                if (!m2_bound) invalidate_cached_buffer(m2_quant.data);
            }
        }
    }

    // Build graph: mul_mat → add(residual) → cpy(back to residual)
    ggml_tensor* contiguous_input = ggml_cont(context.value, input_binding.tensor);
    ggml_tensor* contiguous_residual = ggml_cont(context.value, residual_binding.tensor);

    ggml_tensor* input_2d = (rows == 1)
        ? ggml_reshape_2d(context.value, contiguous_input, in_dim, 1)
        : contiguous_input;

    ggml_tensor* mm = ggml_mul_mat(context.value, m2_binding.tensor, input_2d);
    ggml_tensor* mm_flat = ggml_reshape_1d(context.value, mm, static_cast<int64_t>(rows) * out_dim);
    ggml_tensor* res_flat = ggml_reshape_1d(context.value, contiguous_residual, static_cast<int64_t>(rows) * out_dim);
    ggml_tensor* added = ggml_add(context.value, res_flat, mm_flat);
    ggml_tensor* output_tensor = ggml_cpy(context.value, added, residual_binding.tensor);
    ggml_set_output(output_tensor);

    ggml_cgraph* graph = ggml_new_graph(context.value);
    ggml_build_forward_expand(graph, output_tensor);

    BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
    if (buffer.value == nullptr)
    {
        set_last_error("Failed to allocate backend buffer for fused matmul_add.");
        return 0;
    }

    // Upload data
    if (!use_zero_copy)
    {
        upload_binding(residual_binding, residual_desc.data, residual_binding.raw_bytes);
        if (packed_input.empty())
            upload_binding(input_binding, input_desc.data, input_binding.raw_bytes);
        else
            upload_binding(input_binding, packed_input.data(), input_binding.raw_bytes);
    }

    if (!m2_bound || m2_needs_upload)
        upload_binding(m2_binding, m2_quant.data, m2_binding.raw_bytes);

    ggml_status status = ggml_backend_graph_compute(g_backend, graph);
    if (status != GGML_STATUS_SUCCESS)
    {
        set_last_error("Graph execution failed for fused matmul_add.");
        return 0;
    }
    ggml_backend_synchronize(g_backend);

    if (!use_zero_copy)
        ggml_backend_tensor_get(residual_binding.storage, residual_desc.data, 0, residual_binding.raw_bytes);

    clear_last_error();
    return 1;
}

// ============================================================================
// Fully fused dense SwiGLU FFN with residual add: single GGML graph dispatch.
//
// residual += down_W ^T @ ( silu(gate_part) * up_part ),
//   where [gate_part | up_part] = gate_up_W ^T @ rms_norm(input, normW, eps)
//
// Combines what was previously 3 separate native dispatches:
//   1) FusedRmsNormMatMulQuant  (norm + gate_up matmul)
//   2) SiLUMul / SiLUMulSplit   (activation + multiply)
//   3) FusedMatMulQuantAdd      (down matmul + residual add)
// into a single GGML graph. Saves 2 graph builds, 2 backend allocations and
// 2 host<->backend syncs per FFN per layer per forward call. On Metal this
// dramatically lowers Metal command-buffer overhead which dominates FFN time
// for moderate sequence lengths.
int fused_ffn_swiglu_quant_f32_impl(
    const TensorView2DDesc& residual_desc,
    const TensorView2DDesc& input_desc,
    void* norm_weight_data,
    int norm_weight_count,
    float eps,
    const QuantizedWeightDesc& gate_up_quant,
    const QuantizedWeightDesc& down_quant,
    int half_dim)
{
    if (!ensure_backend())
        return 0;

    if (!validate_desc(residual_desc, "residual") || !validate_desc(input_desc, "input"))
        return 0;

    if (norm_weight_data == nullptr || norm_weight_count <= 0)
    {
        set_last_error("fused_ffn_swiglu: invalid norm weight.");
        return 0;
    }

    if (gate_up_quant.data == nullptr || gate_up_quant.ne0 <= 0 || gate_up_quant.ne1 <= 0 || gate_up_quant.raw_bytes <= 0)
    {
        set_last_error("fused_ffn_swiglu: invalid gate_up weight descriptor.");
        return 0;
    }
    if (down_quant.data == nullptr || down_quant.ne0 <= 0 || down_quant.ne1 <= 0 || down_quant.raw_bytes <= 0)
    {
        set_last_error("fused_ffn_swiglu: invalid down weight descriptor.");
        return 0;
    }

    const int rows = input_desc.dim0;
    const int hidden = input_desc.dim1;
    const int gate_up_out = static_cast<int>(gate_up_quant.ne1);
    const int down_in = static_cast<int>(down_quant.ne0);
    const int down_out = static_cast<int>(down_quant.ne1);

    if (residual_desc.dim0 != rows || residual_desc.dim1 != hidden)
    {
        set_last_error("fused_ffn_swiglu: residual shape mismatch.");
        return 0;
    }
    if (norm_weight_count != hidden)
    {
        set_last_error("fused_ffn_swiglu: norm_weight_count != hidden.");
        return 0;
    }
    if (gate_up_quant.ne0 != hidden)
    {
        set_last_error("fused_ffn_swiglu: gate_up.ne0 != hidden.");
        return 0;
    }
    if (gate_up_out != 2 * half_dim)
    {
        set_last_error("fused_ffn_swiglu: gate_up.ne1 != 2*half_dim.");
        return 0;
    }
    if (down_in != half_dim || down_out != hidden)
    {
        set_last_error("fused_ffn_swiglu: down weight shape mismatch.");
        return 0;
    }

    const std::size_t ctx_size = 4 * 1024 * 1024;
    PooledContextHandle context;
    if (!context.init(ctx_size))
    {
        set_last_error("fused_ffn_swiglu: failed to create ggml context.");
        return 0;
    }

    std::vector<BufferHandle> host_ptr_buffers;
    bool use_zero_copy = can_map_standard_view(input_desc) && can_map_standard_view(residual_desc);

    TensorBinding residual_binding;
    TensorBinding input_binding;
    std::vector<float> packed_input;

    if (use_zero_copy)
    {
        ggml_backend_buffer_t res_buf = nullptr;
        ggml_backend_buffer_t inp_buf = nullptr;
        const bool res_ok = create_binding_from_host_ptr_2d(context.value, g_backend, residual_desc, residual_binding, res_buf);
        const bool inp_ok = res_ok && create_binding_from_host_ptr_2d(context.value, g_backend, input_desc, input_binding, inp_buf);

        if (res_ok && inp_ok)
        {
            host_ptr_buffers.emplace_back(res_buf);
            host_ptr_buffers.emplace_back(inp_buf);
        }
        else
        {
            if (inp_buf != nullptr) ggml_backend_buffer_free(inp_buf);
            if (res_buf != nullptr) ggml_backend_buffer_free(res_buf);
            use_zero_copy = false;
            residual_binding = create_standard_binding(context.value, residual_desc);
            input_binding = can_map_standard_view(input_desc)
                ? create_standard_binding(context.value, input_desc)
                : create_packed_standard_binding(context.value, input_desc, packed_input);
        }
    }
    else
    {
        residual_binding = create_standard_binding(context.value, residual_desc);
        input_binding = can_map_standard_view(input_desc)
            ? create_standard_binding(context.value, input_desc)
            : create_packed_standard_binding(context.value, input_desc, packed_input);
    }

    ggml_tensor* norm_w_tensor = ggml_new_tensor_1d(context.value, GGML_TYPE_F32, norm_weight_count);

    ggml_type gate_up_type = static_cast<ggml_type>(gate_up_quant.ggml_type);
    ggml_tensor* gate_up_tensor = ggml_new_tensor_2d(context.value, gate_up_type, gate_up_quant.ne0, gate_up_quant.ne1);
    TensorBinding gate_up_binding_w = { gate_up_tensor, gate_up_tensor, static_cast<std::size_t>(gate_up_quant.raw_bytes) };

    ggml_type down_type = static_cast<ggml_type>(down_quant.ggml_type);
    ggml_tensor* down_tensor = ggml_new_tensor_2d(context.value, down_type, down_quant.ne0, down_quant.ne1);
    TensorBinding down_binding_w = { down_tensor, down_tensor, static_cast<std::size_t>(down_quant.raw_bytes) };

    if (residual_binding.storage == nullptr || input_binding.storage == nullptr ||
        norm_w_tensor == nullptr || gate_up_tensor == nullptr || down_tensor == nullptr)
    {
        set_last_error("fused_ffn_swiglu: failed to allocate ggml tensors.");
        return 0;
    }

    auto try_cache_quant = [](ggml_tensor* t, const QuantizedWeightDesc& q, bool& bound, bool& needs_upload) {
        bound = false;
        needs_upload = false;
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        if (dev == nullptr || q.raw_bytes < 4096)
            return;
        ggml_backend_buffer_t buf = nullptr;
        void* addr = nullptr;
        if (try_get_cacheable_tensor_buffer(g_backend, dev, t,
                q.data, static_cast<std::size_t>(q.raw_bytes),
                buf, addr, needs_upload))
        {
            ggml_status st = ggml_backend_tensor_alloc(buf, t, addr);
            bound = (st == GGML_STATUS_SUCCESS);
            if (!bound) invalidate_cached_buffer(q.data);
        }
    };

    bool gate_up_bound = false, gate_up_needs_upload = false;
    try_cache_quant(gate_up_tensor, gate_up_quant, gate_up_bound, gate_up_needs_upload);

    bool down_bound = false, down_needs_upload = false;
    try_cache_quant(down_tensor, down_quant, down_bound, down_needs_upload);

    // Norm weight cache.
    bool norm_bound = false, norm_needs_upload = false;
    std::size_t norm_bytes = static_cast<std::size_t>(norm_weight_count) * sizeof(float);
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        if (dev != nullptr && norm_bytes >= 4096)
        {
            ggml_backend_buffer_t buf = nullptr;
            void* addr = nullptr;
            if (try_get_cacheable_tensor_buffer(g_backend, dev, norm_w_tensor,
                    norm_weight_data, norm_bytes,
                    buf, addr, norm_needs_upload))
            {
                ggml_status st = ggml_backend_tensor_alloc(buf, norm_w_tensor, addr);
                norm_bound = (st == GGML_STATUS_SUCCESS);
                if (!norm_bound) invalidate_cached_buffer(norm_weight_data);
            }
        }
    }

    // Build graph.
    ggml_tensor* contiguous_input = ggml_cont(context.value, input_binding.tensor);
    ggml_tensor* contiguous_residual = ggml_cont(context.value, residual_binding.tensor);

    ggml_tensor* normed = ggml_rms_norm(context.value, contiguous_input, eps);
    ggml_tensor* scaled = ggml_mul(context.value, normed, norm_w_tensor);

    ggml_tensor* scaled_2d = (rows == 1)
        ? ggml_reshape_2d(context.value, scaled, hidden, 1)
        : scaled;

    // gate_up = scaled @ gate_up_W^T -> ggml semantics: ne0=gate_up_out, ne1=rows
    ggml_tensor* gate_up_mm = ggml_mul_mat(context.value, gate_up_binding_w.tensor, scaled_2d);

    const std::size_t gu_row_bytes = static_cast<std::size_t>(gate_up_out) * sizeof(float);
    const std::size_t half_bytes = static_cast<std::size_t>(half_dim) * sizeof(float);

    ggml_tensor* gate_view = ggml_view_2d(context.value, gate_up_mm,
        half_dim, rows, gu_row_bytes, 0);
    ggml_tensor* up_view = ggml_view_2d(context.value, gate_up_mm,
        half_dim, rows, gu_row_bytes, half_bytes);

    ggml_tensor* gate_cont = ggml_cont(context.value, gate_view);
    ggml_tensor* up_cont = ggml_cont(context.value, up_view);

    ggml_tensor* silu_gate = ggml_silu(context.value, gate_cont);
    ggml_tensor* swiglu = ggml_mul(context.value, silu_gate, up_cont);

    ggml_tensor* swiglu_2d = (rows == 1)
        ? ggml_reshape_2d(context.value, swiglu, half_dim, 1)
        : swiglu;

    ggml_tensor* down_mm = ggml_mul_mat(context.value, down_binding_w.tensor, swiglu_2d);

    ggml_tensor* down_flat = ggml_reshape_1d(context.value, down_mm, static_cast<int64_t>(rows) * hidden);
    ggml_tensor* res_flat = ggml_reshape_1d(context.value, contiguous_residual, static_cast<int64_t>(rows) * hidden);
    ggml_tensor* added = ggml_add(context.value, res_flat, down_flat);

    ggml_tensor* output_tensor = ggml_cpy(context.value, added, residual_binding.tensor);
    if (output_tensor == nullptr)
    {
        set_last_error("fused_ffn_swiglu: failed to create output cpy node.");
        return 0;
    }

    ggml_set_output(output_tensor);

    ggml_cgraph* graph = ggml_new_graph(context.value);
    if (graph == nullptr)
    {
        set_last_error("fused_ffn_swiglu: failed to create graph.");
        return 0;
    }
    ggml_build_forward_expand(graph, output_tensor);

    BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
    if (buffer.value == nullptr)
    {
        set_last_error("fused_ffn_swiglu: failed to allocate backend buffer.");
        return 0;
    }

    if (!use_zero_copy)
    {
        upload_binding(residual_binding, residual_desc.data, residual_binding.raw_bytes);
        if (packed_input.empty())
            upload_binding(input_binding, input_desc.data, input_binding.raw_bytes);
        else
            upload_binding(input_binding, packed_input.data(), input_binding.raw_bytes);
    }

    if (!gate_up_bound || gate_up_needs_upload)
        upload_binding(gate_up_binding_w, gate_up_quant.data, gate_up_binding_w.raw_bytes);
    if (!down_bound || down_needs_upload)
        upload_binding(down_binding_w, down_quant.data, down_binding_w.raw_bytes);
    if (!norm_bound || norm_needs_upload)
    {
        TensorBinding tmp = { norm_w_tensor, norm_w_tensor, norm_bytes };
        upload_binding(tmp, norm_weight_data, norm_bytes);
    }

    ggml_status status = ggml_backend_graph_compute(g_backend, graph);
    if (status != GGML_STATUS_SUCCESS)
    {
        set_last_error("fused_ffn_swiglu: graph execution failed.");
        return 0;
    }
    ggml_backend_synchronize(g_backend);

    if (!use_zero_copy)
        ggml_backend_tensor_get(residual_binding.storage, residual_desc.data, 0, residual_binding.raw_bytes);

    clear_last_error();
    return 1;
}

// ============================================================================
// Fused vision encoder block: runs the entire attention sub-block (minus RoPE)
// and the entire MLP sub-block as ONE Metal graph dispatch instead of ~14
// separate dispatches. For 27 encoder blocks this eliminates ~350 Metal command
// buffer round-trips, cutting the vision encoder time by an order of magnitude.
//
// Graph topology:
//   ln1 -> qkv_matmul+bias -> [output: split qkv for CPU RoPE + SDPA]
//   attn_in (post-SDPA) -> out_proj+bias -> residual1
//   ln2 -> up_matmul+bias -> GELU -> down_matmul+bias -> residual2
// ============================================================================
int fused_vision_mlp_f32_impl(
    const TensorView2DDesc& hidden_desc,   // [N, D] in/out (residual is in-place)
    const float* ln_w, const float* ln_b, int ln_dim, float eps,
    const float* up_w_data,   int up_ne0, int up_ne1, std::size_t up_bytes,   // [D, Dff] transposed already
    const float* up_b_data,   int up_b_dim,
    const float* down_w_data, int down_ne0, int down_ne1, std::size_t down_bytes,
    const float* down_b_data, int down_b_dim)
{
    if (!ensure_backend())
        return 0;
    if (!validate_desc(hidden_desc, "hidden"))
        return 0;

    const int rows = hidden_desc.dim0;   // numPatches
    const int hidden = hidden_desc.dim1; // hiddenSize
    const int dff = up_ne1;              // intermediate_size

    const std::size_t ctx_size = 4 * 1024 * 1024;
    PooledContextHandle context;
    if (!context.init(ctx_size))
    {
        set_last_error("fused_vision_mlp: context init failed.");
        return 0;
    }

    std::vector<BufferHandle> host_ptr_buffers;
    bool use_zero_copy = can_map_standard_view(hidden_desc);

    TensorBinding hidden_binding;
    if (use_zero_copy)
    {
        ggml_backend_buffer_t buf = nullptr;
        if (!create_binding_from_host_ptr_2d(context.value, g_backend, hidden_desc, hidden_binding, buf))
        {
            use_zero_copy = false;
            hidden_binding = create_standard_binding(context.value, hidden_desc);
        }
        else
            host_ptr_buffers.emplace_back(buf);
    }
    else
        hidden_binding = create_standard_binding(context.value, hidden_desc);

    ggml_tensor* ln_w_t  = ggml_new_tensor_1d(context.value, GGML_TYPE_F32, ln_dim);
    ggml_tensor* ln_b_t  = ggml_new_tensor_1d(context.value, GGML_TYPE_F32, ln_dim);
    ggml_tensor* up_w_t  = ggml_new_tensor_2d(context.value, GGML_TYPE_F32, up_ne0, up_ne1);
    ggml_tensor* up_b_t  = ggml_new_tensor_1d(context.value, GGML_TYPE_F32, up_b_dim);
    ggml_tensor* down_w_t = ggml_new_tensor_2d(context.value, GGML_TYPE_F32, down_ne0, down_ne1);
    ggml_tensor* down_b_t = ggml_new_tensor_1d(context.value, GGML_TYPE_F32, down_b_dim);

    if (!hidden_binding.storage || !ln_w_t || !ln_b_t || !up_w_t || !up_b_t || !down_w_t || !down_b_t)
    {
        set_last_error("fused_vision_mlp: tensor alloc failed.");
        return 0;
    }

    // Pre-bind cacheable weight tensors BEFORE graph allocation so that
    // ggml_backend_alloc_ctx_tensors skips already-bound tensors.
    auto try_cache_weight = [&](ggml_tensor* t, const void* data, std::size_t bytes, bool& bound, bool& needs_upload) {
        bound = false;
        needs_upload = false;
        void* mutable_data = const_cast<void*>(data);
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        if (dev && bytes >= 4096)
        {
            ggml_backend_buffer_t buf = nullptr;
            void* addr = nullptr;
            if (try_get_cacheable_tensor_buffer(g_backend, dev, t, mutable_data, bytes, buf, addr, needs_upload))
            {
                ggml_status st = ggml_backend_tensor_alloc(buf, t, addr);
                bound = (st == GGML_STATUS_SUCCESS);
                if (!bound) invalidate_cached_buffer(mutable_data);
            }
        }
    };

    bool ln_w_bound = false, ln_w_upload = false;
    bool ln_b_bound = false, ln_b_upload = false;
    bool up_w_bound = false, up_w_upload = false;
    bool up_b_bound = false, up_b_upload = false;
    bool down_w_bound = false, down_w_upload = false;
    bool down_b_bound = false, down_b_upload = false;

    try_cache_weight(ln_w_t,   ln_w,       ln_dim * sizeof(float),       ln_w_bound, ln_w_upload);
    try_cache_weight(ln_b_t,   ln_b,       ln_dim * sizeof(float),       ln_b_bound, ln_b_upload);
    try_cache_weight(up_w_t,   up_w_data,  up_bytes,                     up_w_bound, up_w_upload);
    try_cache_weight(up_b_t,   up_b_data,  up_b_dim * sizeof(float),     up_b_bound, up_b_upload);
    try_cache_weight(down_w_t, down_w_data, down_bytes,                  down_w_bound, down_w_upload);
    try_cache_weight(down_b_t, down_b_data, down_b_dim * sizeof(float),  down_b_bound, down_b_upload);

    // Build the computation graph.
    ggml_context* ctx = context.value;
    ggml_tensor* inp = ggml_cont(ctx, hidden_binding.tensor);

    // LayerNorm: mean-subtract, variance-normalize, scale+shift
    ggml_tensor* normed = ggml_norm(ctx, inp, eps);
    ggml_tensor* scaled = ggml_mul(ctx, normed, ln_w_t);
    ggml_tensor* ln_out = ggml_add(ctx, scaled, ln_b_t);

    // Up projection: ln_out @ up_w^T + up_bias
    ggml_tensor* ln_2d = (rows == 1) ? ggml_reshape_2d(ctx, ln_out, hidden, 1) : ln_out;
    ggml_tensor* fc1 = ggml_mul_mat(ctx, up_w_t, ln_2d);
    ggml_tensor* fc1_bias = ggml_add(ctx, fc1, ggml_repeat(ctx, ggml_reshape_2d(ctx, up_b_t, dff, 1), fc1));

    // GELU activation
    ggml_tensor* fc1_gelu = ggml_gelu(ctx, fc1_bias);

    // Down projection: fc1_gelu @ down_w^T + down_bias
    ggml_tensor* fc1_2d = (rows == 1) ? ggml_reshape_2d(ctx, fc1_gelu, dff, 1) : fc1_gelu;
    ggml_tensor* fc2 = ggml_mul_mat(ctx, down_w_t, fc1_2d);
    ggml_tensor* fc2_bias = ggml_add(ctx, fc2, ggml_repeat(ctx, ggml_reshape_2d(ctx, down_b_t, hidden, 1), fc2));

    // Residual add: hidden += fc2_bias
    ggml_tensor* res_flat = ggml_reshape_1d(ctx, inp, static_cast<int64_t>(rows) * hidden);
    ggml_tensor* fc2_flat = ggml_reshape_1d(ctx, fc2_bias, static_cast<int64_t>(rows) * hidden);
    ggml_tensor* added = ggml_add(ctx, res_flat, fc2_flat);

    ggml_tensor* output = ggml_cpy(ctx, added, hidden_binding.tensor);
    if (!output)
    {
        set_last_error("fused_vision_mlp: output cpy failed.");
        return 0;
    }
    ggml_set_output(output);

    ggml_cgraph* graph = ggml_new_graph(ctx);
    if (!graph)
    {
        set_last_error("fused_vision_mlp: graph creation failed.");
        return 0;
    }
    ggml_build_forward_expand(graph, output);

    BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
    if (!buffer.value)
    {
        set_last_error("fused_vision_mlp: buffer alloc failed.");
        return 0;
    }

    // Upload data.
    if (!use_zero_copy)
        upload_binding(hidden_binding, hidden_desc.data, hidden_binding.raw_bytes);

    auto upload_if_needed = [](ggml_tensor* t, const void* data, std::size_t bytes, bool bound, bool needs_upload) {
        if (!bound || needs_upload)
            ggml_backend_tensor_set(t, data, 0, bytes);
    };
    upload_if_needed(ln_w_t,   ln_w,       ln_dim * sizeof(float),      ln_w_bound, ln_w_upload);
    upload_if_needed(ln_b_t,   ln_b,       ln_dim * sizeof(float),      ln_b_bound, ln_b_upload);
    upload_if_needed(up_w_t,   up_w_data,  up_bytes,                    up_w_bound, up_w_upload);
    upload_if_needed(up_b_t,   up_b_data,  up_b_dim * sizeof(float),    up_b_bound, up_b_upload);
    upload_if_needed(down_w_t, down_w_data, down_bytes,                 down_w_bound, down_w_upload);
    upload_if_needed(down_b_t, down_b_data, down_b_dim * sizeof(float), down_b_bound, down_b_upload);

    ggml_status status = ggml_backend_graph_compute(g_backend, graph);
    if (status != GGML_STATUS_SUCCESS)
    {
        set_last_error("fused_vision_mlp: graph compute failed.");
        return 0;
    }
    ggml_backend_synchronize(g_backend);

    if (!use_zero_copy)
        ggml_backend_tensor_get(hidden_binding.storage, hidden_desc.data, 0, hidden_binding.raw_bytes);

    clear_last_error();
    return 1;
}

// ============================================================================
// Fused vision attention: LN + QKV + bias + RoPE (cos/sin tables) + SDPA +
// output proj + bias + residual. All in one GGML graph dispatch.
// ============================================================================
int fused_vision_attention_f32_impl(
    const TensorView2DDesc& hidden_desc,  // [N, D] in/out
    const float* ln_w, const float* ln_b, int ln_dim, float eps,
    const float* qkv_w_data, int qkv_ne0, int qkv_ne1, std::size_t qkv_bytes,
    const float* qkv_b_data, int qkv_b_dim,
    const float* out_w_data, int out_ne0, int out_ne1, std::size_t out_bytes,
    const float* out_b_data, int out_b_dim,
    const float* cos_table, const float* sin_table,
    int num_patches, int num_heads, int head_dim, int half_dim,
    float attn_scale)
{
    if (!ensure_backend())
        return 0;
    if (!validate_desc(hidden_desc, "hidden"))
        return 0;

    const int rows = hidden_desc.dim0;
    const int hidden = hidden_desc.dim1;
    const int triple_hidden = 3 * hidden;

    const std::size_t ctx_size = 8 * 1024 * 1024;
    PooledContextHandle context;
    if (!context.init(ctx_size))
    {
        set_last_error("fused_vision_attn: context init failed.");
        return 0;
    }

    std::vector<BufferHandle> host_ptr_buffers;
    bool use_zero_copy = can_map_standard_view(hidden_desc);

    TensorBinding hidden_binding;
    if (use_zero_copy)
    {
        ggml_backend_buffer_t buf = nullptr;
        if (!create_binding_from_host_ptr_2d(context.value, g_backend, hidden_desc, hidden_binding, buf))
        {
            use_zero_copy = false;
            hidden_binding = create_standard_binding(context.value, hidden_desc);
        }
        else
            host_ptr_buffers.emplace_back(buf);
    }
    else
        hidden_binding = create_standard_binding(context.value, hidden_desc);

    ggml_context* ctx = context.value;

    ggml_tensor* ln_w_t   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ln_dim);
    ggml_tensor* ln_b_t   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ln_dim);
    ggml_tensor* qkv_w_t  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, qkv_ne0, qkv_ne1);
    ggml_tensor* qkv_b_t  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, qkv_b_dim);
    ggml_tensor* out_w_t  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, out_ne0, out_ne1);
    ggml_tensor* out_b_t  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_b_dim);

    int cos_sin_elems = num_patches * half_dim;
    ggml_tensor* cos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cos_sin_elems);
    ggml_tensor* sin_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cos_sin_elems);

    if (!hidden_binding.storage || !ln_w_t || !ln_b_t || !qkv_w_t || !qkv_b_t || !out_w_t || !out_b_t || !cos_t || !sin_t)
    {
        set_last_error("fused_vision_attn: tensor alloc failed.");
        return 0;
    }

    // Pre-bind cacheable weights.
    auto try_bind = [&](ggml_tensor* t, const void* data, std::size_t bytes, bool& bound, bool& needs_upload) {
        bound = false; needs_upload = false;
        void* md = const_cast<void*>(data);
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        if (dev && bytes >= 4096)
        {
            ggml_backend_buffer_t buf = nullptr; void* addr = nullptr;
            if (try_get_cacheable_tensor_buffer(g_backend, dev, t, md, bytes, buf, addr, needs_upload))
            { ggml_status st = ggml_backend_tensor_alloc(buf, t, addr); bound = (st == GGML_STATUS_SUCCESS); if (!bound) invalidate_cached_buffer(md); }
        }
    };

    bool b1, u1, b2, u2, b3, u3, b4, u4, b5, u5, b6, u6;
    try_bind(ln_w_t,  ln_w,       ln_dim * sizeof(float),      b1, u1);
    try_bind(ln_b_t,  ln_b,       ln_dim * sizeof(float),      b2, u2);
    try_bind(qkv_w_t, qkv_w_data, qkv_bytes,                  b3, u3);
    try_bind(qkv_b_t, qkv_b_data, qkv_b_dim * sizeof(float),  b4, u4);
    try_bind(out_w_t, out_w_data, out_bytes,                   b5, u5);
    try_bind(out_b_t, out_b_data, out_b_dim * sizeof(float),   b6, u6);

    // Build graph.
    ggml_tensor* inp = ggml_cont(ctx, hidden_binding.tensor);

    // LayerNorm
    ggml_tensor* normed = ggml_norm(ctx, inp, eps);
    ggml_tensor* ln_scaled = ggml_mul(ctx, normed, ln_w_t);
    ggml_tensor* ln_out = ggml_add(ctx, ln_scaled, ln_b_t);

    // QKV projection + bias
    ggml_tensor* ln_2d = (rows == 1) ? ggml_reshape_2d(ctx, ln_out, hidden, 1) : ln_out;
    ggml_tensor* qkv = ggml_mul_mat(ctx, qkv_w_t, ln_2d);
    ggml_tensor* qkv_biased = ggml_add(ctx, qkv, ggml_repeat(ctx, ggml_reshape_2d(ctx, qkv_b_t, triple_hidden, 1), qkv));

    // Split Q, K, V: [N, 3*D] -> three [N, D]
    std::size_t row_bytes = static_cast<std::size_t>(triple_hidden) * sizeof(float);
    std::size_t d_bytes = static_cast<std::size_t>(hidden) * sizeof(float);
    ggml_tensor* q_raw = ggml_cont(ctx, ggml_view_2d(ctx, qkv_biased, hidden, rows, row_bytes, 0));
    ggml_tensor* k_raw = ggml_cont(ctx, ggml_view_2d(ctx, qkv_biased, hidden, rows, row_bytes, d_bytes));
    ggml_tensor* v_raw = ggml_cont(ctx, ggml_view_2d(ctx, qkv_biased, hidden, rows, row_bytes, 2 * d_bytes));

    // Apply RoPE via cos/sin tables: NeoX rotation
    // Q/K shape: [hidden, N] in GGML = [N, hidden] in row-major
    // Reshape to [half_dim, 2, num_heads, N] to split lo/hi halves
    // Actually simpler: reshape to [head_dim, num_heads, N], then split
    ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_raw, head_dim, num_heads, rows);
    ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_raw, head_dim, num_heads, rows);

    // cos/sin tables: [cos_sin_elems] -> reshape to [half_dim, 1, N] for broadcasting
    ggml_tensor* cos_3d = ggml_reshape_3d(ctx, cos_t, half_dim, 1, rows);
    ggml_tensor* sin_3d = ggml_reshape_3d(ctx, sin_t, half_dim, 1, rows);

    // Split Q into lo/hi halves along dim0 (head_dim -> half_dim + half_dim)
    std::size_t head_row_bytes = static_cast<std::size_t>(head_dim) * sizeof(float);
    std::size_t half_bytes_local = static_cast<std::size_t>(half_dim) * sizeof(float);

    auto apply_rope = [&](ggml_tensor* x_3d) -> ggml_tensor* {
        // x_3d: [head_dim, num_heads, N]
        ggml_tensor* x_lo = ggml_view_3d(ctx, x_3d, half_dim, num_heads, rows,
            head_row_bytes, head_row_bytes * num_heads, 0);
        ggml_tensor* x_hi = ggml_view_3d(ctx, x_3d, half_dim, num_heads, rows,
            head_row_bytes, head_row_bytes * num_heads, half_bytes_local);
        ggml_tensor* lo_c = ggml_cont(ctx, x_lo);
        ggml_tensor* hi_c = ggml_cont(ctx, x_hi);
        // out_lo = lo * cos - hi * sin
        ggml_tensor* lo_cos = ggml_mul(ctx, lo_c, cos_3d);
        ggml_tensor* hi_sin = ggml_mul(ctx, hi_c, sin_3d);
        ggml_tensor* out_lo = ggml_sub(ctx, lo_cos, hi_sin);
        // out_hi = lo * sin + hi * cos
        ggml_tensor* lo_sin = ggml_mul(ctx, lo_c, sin_3d);
        ggml_tensor* hi_cos = ggml_mul(ctx, hi_c, cos_3d);
        ggml_tensor* out_hi = ggml_add(ctx, lo_sin, hi_cos);
        // Concat along dim0: [half_dim, H, N] + [half_dim, H, N] -> [head_dim, H, N]
        return ggml_concat(ctx, out_lo, out_hi, 0);
    };

    ggml_tensor* q_roped = apply_rope(q_3d);
    ggml_tensor* k_roped = apply_rope(k_3d);
    ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_raw, head_dim, num_heads, rows);

    // SDPA: flash_attn_ext expects Q=[head_dim, N, H], K=[head_dim, N, H_kv], V=[head_dim, N, H_kv]
    ggml_tensor* q_perm = ggml_permute(ctx, q_roped, 0, 2, 1, 3); // [head_dim, N, H]
    ggml_tensor* k_perm = ggml_permute(ctx, k_roped, 0, 2, 1, 3);
    ggml_tensor* v_perm = ggml_permute(ctx, v_3d, 0, 2, 1, 3);

    ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q_perm, k_perm, v_perm, nullptr, attn_scale, 0.0f, 0.0f);
    // attn_out: [head_dim, N, H] -> reshape to [hidden, N] -> [N, hidden]
    ggml_tensor* attn_flat = ggml_reshape_2d(ctx, ggml_cont(ctx, attn_out), hidden, rows);

    // Output projection + bias
    ggml_tensor* out_proj = ggml_mul_mat(ctx, out_w_t, attn_flat);
    ggml_tensor* out_biased = ggml_add(ctx, out_proj, ggml_repeat(ctx, ggml_reshape_2d(ctx, out_b_t, hidden, 1), out_proj));

    // Residual add
    ggml_tensor* res_flat = ggml_reshape_1d(ctx, inp, static_cast<int64_t>(rows) * hidden);
    ggml_tensor* out_flat = ggml_reshape_1d(ctx, out_biased, static_cast<int64_t>(rows) * hidden);
    ggml_tensor* added = ggml_add(ctx, res_flat, out_flat);

    ggml_tensor* output = ggml_cpy(ctx, added, hidden_binding.tensor);
    if (!output)
    {
        set_last_error("fused_vision_attn: output cpy failed.");
        return 0;
    }
    ggml_set_output(output);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 4096, false);
    if (!graph) { set_last_error("fused_vision_attn: graph failed."); return 0; }
    ggml_build_forward_expand(graph, output);

    BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
    if (!buffer.value) { set_last_error("fused_vision_attn: buffer alloc failed."); return 0; }

    if (!use_zero_copy)
        upload_binding(hidden_binding, hidden_desc.data, hidden_binding.raw_bytes);

    auto up_if = [](ggml_tensor* t, const void* d, std::size_t s, bool bound, bool need) { if (!bound || need) ggml_backend_tensor_set(t, d, 0, s); };
    up_if(ln_w_t,  ln_w,       ln_dim * sizeof(float),      b1, u1);
    up_if(ln_b_t,  ln_b,       ln_dim * sizeof(float),      b2, u2);
    up_if(qkv_w_t, qkv_w_data, qkv_bytes,                  b3, u3);
    up_if(qkv_b_t, qkv_b_data, qkv_b_dim * sizeof(float),  b4, u4);
    up_if(out_w_t, out_w_data, out_bytes,                   b5, u5);
    up_if(out_b_t, out_b_data, out_b_dim * sizeof(float),   b6, u6);
    ggml_backend_tensor_set(cos_t, cos_table, 0, cos_sin_elems * sizeof(float));
    ggml_backend_tensor_set(sin_t, sin_table, 0, cos_sin_elems * sizeof(float));

    ggml_status status = ggml_backend_graph_compute(g_backend, graph);
    if (status != GGML_STATUS_SUCCESS) { set_last_error("fused_vision_attn: compute failed."); return 0; }
    ggml_backend_synchronize(g_backend);

    if (!use_zero_copy)
        ggml_backend_tensor_get(hidden_binding.storage, hidden_desc.data, 0, hidden_binding.raw_bytes);

    clear_last_error();
    return 1;
}

// ============================================================================
// Fused output projection + residual + FFN (RMSNorm + SwiGLU) in one graph.
// Combines what was previously TryLinearAddInto + FusedFFNSwiGLUQuant into a
// single Metal command buffer. Applied to every layer (32), this saves 32
// dispatch round-trips from the text model prefill hot path.
//
// Graph: residual += matmul(input, outW)                    [output proj]
//        normed   = rms_norm(residual) * ffnNormW           [pre-FFN norm]
//        gate_up  = matmul(normed, gateUpW)                 [FFN gate+up]
//        swiglu   = silu(gate_up[:H]) * gate_up[H:]         [activation]
//        residual += matmul(swiglu, downW)                   [FFN down + residual]
// ============================================================================
int fused_outproj_ffn_quant_f32_impl(
    const TensorView2DDesc& residual_desc,
    const TensorView2DDesc& input_desc,
    const QuantizedWeightDesc& out_proj_quant,
    void* ffn_norm_weight_data, int ffn_norm_count, float eps,
    const QuantizedWeightDesc& gate_up_quant,
    const QuantizedWeightDesc& down_quant,
    int half_dim)
{
    if (!ensure_backend()) return 0;
    if (!validate_desc(residual_desc, "residual") || !validate_desc(input_desc, "input")) return 0;

    const int rows = input_desc.dim0;
    const int hidden = residual_desc.dim1;
    const int gate_up_out = static_cast<int>(gate_up_quant.ne1);

    const std::size_t ctx_size = 4 * 1024 * 1024;
    PooledContextHandle context;
    if (!context.init(ctx_size))
    { set_last_error("fused_outproj_ffn: context failed."); return 0; }

    std::vector<BufferHandle> host_ptr_buffers;
    bool use_zero_copy = can_map_standard_view(input_desc) && can_map_standard_view(residual_desc);

    TensorBinding residual_binding, input_binding;
    std::vector<float> packed_input;

    if (use_zero_copy)
    {
        ggml_backend_buffer_t res_buf = nullptr, inp_buf = nullptr;
        bool res_ok = create_binding_from_host_ptr_2d(context.value, g_backend, residual_desc, residual_binding, res_buf);
        bool inp_ok = res_ok && create_binding_from_host_ptr_2d(context.value, g_backend, input_desc, input_binding, inp_buf);
        if (res_ok && inp_ok) { host_ptr_buffers.emplace_back(res_buf); host_ptr_buffers.emplace_back(inp_buf); }
        else {
            if (inp_buf) ggml_backend_buffer_free(inp_buf);
            if (res_buf) ggml_backend_buffer_free(res_buf);
            use_zero_copy = false;
            residual_binding = create_standard_binding(context.value, residual_desc);
            input_binding = can_map_standard_view(input_desc)
                ? create_standard_binding(context.value, input_desc)
                : create_packed_standard_binding(context.value, input_desc, packed_input);
        }
    }
    else
    {
        residual_binding = create_standard_binding(context.value, residual_desc);
        input_binding = can_map_standard_view(input_desc)
            ? create_standard_binding(context.value, input_desc)
            : create_packed_standard_binding(context.value, input_desc, packed_input);
    }

    ggml_context* ctx = context.value;

    ggml_type out_type = static_cast<ggml_type>(out_proj_quant.ggml_type);
    ggml_tensor* out_w = ggml_new_tensor_2d(ctx, out_type, out_proj_quant.ne0, out_proj_quant.ne1);
    TensorBinding out_w_bind = { out_w, out_w, static_cast<std::size_t>(out_proj_quant.raw_bytes) };

    ggml_tensor* norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ffn_norm_count);

    ggml_type gu_type = static_cast<ggml_type>(gate_up_quant.ggml_type);
    ggml_tensor* gu_w = ggml_new_tensor_2d(ctx, gu_type, gate_up_quant.ne0, gate_up_quant.ne1);
    TensorBinding gu_w_bind = { gu_w, gu_w, static_cast<std::size_t>(gate_up_quant.raw_bytes) };

    ggml_type dn_type = static_cast<ggml_type>(down_quant.ggml_type);
    ggml_tensor* dn_w = ggml_new_tensor_2d(ctx, dn_type, down_quant.ne0, down_quant.ne1);
    TensorBinding dn_w_bind = { dn_w, dn_w, static_cast<std::size_t>(down_quant.raw_bytes) };

    if (!residual_binding.storage || !input_binding.storage || !out_w || !norm_w || !gu_w || !dn_w)
    { set_last_error("fused_outproj_ffn: tensor alloc failed."); return 0; }

    // Pre-bind cacheable weights.
    auto bind_cached = [](ggml_tensor* t, const QuantizedWeightDesc& q, bool& bound, bool& needs_upload) {
        bound = false; needs_upload = false;
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        if (!dev || q.raw_bytes < 4096) return;
        ggml_backend_buffer_t buf = nullptr; void* addr = nullptr;
        if (try_get_cacheable_tensor_buffer(g_backend, dev, t, q.data, static_cast<std::size_t>(q.raw_bytes), buf, addr, needs_upload))
        { ggml_status st = ggml_backend_tensor_alloc(buf, t, addr); bound = (st == GGML_STATUS_SUCCESS); if (!bound) invalidate_cached_buffer(q.data); }
    };

    bool out_bound, out_upl, gu_bound, gu_upl, dn_bound, dn_upl;
    bind_cached(out_w, out_proj_quant, out_bound, out_upl);
    bind_cached(gu_w, gate_up_quant, gu_bound, gu_upl);
    bind_cached(dn_w, down_quant, dn_bound, dn_upl);

    bool norm_bound = false, norm_upl = false;
    std::size_t norm_bytes = static_cast<std::size_t>(ffn_norm_count) * sizeof(float);
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        if (dev && norm_bytes >= 4096) {
            ggml_backend_buffer_t buf = nullptr; void* addr = nullptr;
            if (try_get_cacheable_tensor_buffer(g_backend, dev, norm_w, ffn_norm_weight_data, norm_bytes, buf, addr, norm_upl))
            { ggml_status st = ggml_backend_tensor_alloc(buf, norm_w, addr); norm_bound = (st == GGML_STATUS_SUCCESS); if (!norm_bound) invalidate_cached_buffer(ffn_norm_weight_data); }
        }
    }

    // Build graph.
    ggml_tensor* cont_input = ggml_cont(ctx, input_binding.tensor);
    ggml_tensor* cont_res   = ggml_cont(ctx, residual_binding.tensor);

    // Phase 1: output projection + residual
    ggml_tensor* inp_2d = (rows == 1) ? ggml_reshape_2d(ctx, cont_input, input_desc.dim1, 1) : cont_input;
    ggml_tensor* out_mm = ggml_mul_mat(ctx, out_w, inp_2d);
    ggml_tensor* out_flat = ggml_reshape_1d(ctx, out_mm, static_cast<int64_t>(rows) * hidden);
    ggml_tensor* res_flat1 = ggml_reshape_1d(ctx, cont_res, static_cast<int64_t>(rows) * hidden);
    ggml_tensor* res_plus_out = ggml_add(ctx, res_flat1, out_flat);
    ggml_tensor* res_2d = ggml_reshape_2d(ctx, res_plus_out, hidden, rows);

    // Phase 2: RMSNorm + FFN SwiGLU + residual
    ggml_tensor* normed = ggml_rms_norm(ctx, res_2d, eps);
    ggml_tensor* scaled = ggml_mul(ctx, normed, norm_w);
    ggml_tensor* scaled_2d = (rows == 1) ? ggml_reshape_2d(ctx, scaled, hidden, 1) : scaled;

    ggml_tensor* gu_mm = ggml_mul_mat(ctx, gu_w, scaled_2d);
    std::size_t gu_row_bytes = static_cast<std::size_t>(gate_up_out) * sizeof(float);
    std::size_t half_bytes = static_cast<std::size_t>(half_dim) * sizeof(float);
    ggml_tensor* gate_v = ggml_cont(ctx, ggml_view_2d(ctx, gu_mm, half_dim, rows, gu_row_bytes, 0));
    ggml_tensor* up_v   = ggml_cont(ctx, ggml_view_2d(ctx, gu_mm, half_dim, rows, gu_row_bytes, half_bytes));
    ggml_tensor* silu_gate = ggml_silu(ctx, gate_v);
    ggml_tensor* swiglu = ggml_mul(ctx, silu_gate, up_v);
    ggml_tensor* swiglu_2d = (rows == 1) ? ggml_reshape_2d(ctx, swiglu, half_dim, 1) : swiglu;
    ggml_tensor* dn_mm = ggml_mul_mat(ctx, dn_w, swiglu_2d);

    ggml_tensor* dn_flat = ggml_reshape_1d(ctx, dn_mm, static_cast<int64_t>(rows) * hidden);
    ggml_tensor* res_flat2 = ggml_reshape_1d(ctx, res_2d, static_cast<int64_t>(rows) * hidden);
    ggml_tensor* final_res = ggml_add(ctx, res_flat2, dn_flat);

    ggml_tensor* output = ggml_cpy(ctx, final_res, residual_binding.tensor);
    if (!output) { set_last_error("fused_outproj_ffn: output cpy failed."); return 0; }
    ggml_set_output(output);

    ggml_cgraph* graph = ggml_new_graph(ctx);
    if (!graph) { set_last_error("fused_outproj_ffn: graph failed."); return 0; }
    ggml_build_forward_expand(graph, output);

    BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
    if (!buffer.value) { set_last_error("fused_outproj_ffn: buffer alloc failed."); return 0; }

    if (!use_zero_copy) {
        upload_binding(residual_binding, residual_desc.data, residual_binding.raw_bytes);
        if (packed_input.empty()) upload_binding(input_binding, input_desc.data, input_binding.raw_bytes);
        else upload_binding(input_binding, packed_input.data(), input_binding.raw_bytes);
    }

    if (!out_bound || out_upl) upload_binding(out_w_bind, out_proj_quant.data, out_w_bind.raw_bytes);
    if (!gu_bound || gu_upl) upload_binding(gu_w_bind, gate_up_quant.data, gu_w_bind.raw_bytes);
    if (!dn_bound || dn_upl) upload_binding(dn_w_bind, down_quant.data, dn_w_bind.raw_bytes);
    if (!norm_bound || norm_upl) { TensorBinding tmp = { norm_w, norm_w, norm_bytes }; upload_binding(tmp, ffn_norm_weight_data, norm_bytes); }

    ggml_status status = ggml_backend_graph_compute(g_backend, graph);
    if (status != GGML_STATUS_SUCCESS) { set_last_error("fused_outproj_ffn: compute failed."); return 0; }
    ggml_backend_synchronize(g_backend);

    if (!use_zero_copy)
        ggml_backend_tensor_get(residual_binding.storage, residual_desc.data, 0, residual_binding.raw_bytes);

    clear_last_error();
    return 1;
}

// ============================================================================
// Fused output projection + residual + RMSNorm + router projection for MoE
// decode. Combines 3 separate dispatches (outproj+add, norm, router) into 1.
// Outputs: residual (updated in-place), normedOut (for MoE expert input),
//          routerOut (for CPU top-K routing).
// ============================================================================
int fused_outproj_norm_router_quant_f32_impl(
    const TensorView2DDesc& residual_desc,
    const TensorView2DDesc& input_desc,
    const QuantizedWeightDesc& out_proj_quant,
    void* norm_weight_data, int norm_count, float eps,
    const TensorView2DDesc& normed_out_desc,
    const QuantizedWeightDesc& router_quant,
    const TensorView2DDesc& router_out_desc)
{
    if (!ensure_backend()) return 0;
    if (!validate_desc(residual_desc, "residual") || !validate_desc(input_desc, "input")
        || !validate_desc(normed_out_desc, "normed_out") || !validate_desc(router_out_desc, "router_out"))
        return 0;

    const int rows = input_desc.dim0;
    const int hidden = residual_desc.dim1;

    const std::size_t ctx_size = 4 * 1024 * 1024;
    PooledContextHandle context;
    if (!context.init(ctx_size))
    { set_last_error("fused_outproj_norm_router: context failed."); return 0; }

    std::vector<BufferHandle> host_ptr_buffers;
    bool use_zero_copy = can_map_standard_view(residual_desc) && can_map_standard_view(input_desc)
        && can_map_standard_view(normed_out_desc) && can_map_standard_view(router_out_desc);

    TensorBinding residual_binding, input_binding, normed_binding, router_binding;
    std::vector<float> packed_input;

    if (use_zero_copy)
    {
        ggml_backend_buffer_t rb = nullptr, ib = nullptr, nb = nullptr, rtb = nullptr;
        bool ok = create_binding_from_host_ptr_2d(context.value, g_backend, residual_desc, residual_binding, rb);
        ok = ok && create_binding_from_host_ptr_2d(context.value, g_backend, input_desc, input_binding, ib);
        ok = ok && create_binding_from_host_ptr_2d(context.value, g_backend, normed_out_desc, normed_binding, nb);
        ok = ok && create_binding_from_host_ptr_2d(context.value, g_backend, router_out_desc, router_binding, rtb);
        if (ok) {
            host_ptr_buffers.emplace_back(rb); host_ptr_buffers.emplace_back(ib);
            host_ptr_buffers.emplace_back(nb); host_ptr_buffers.emplace_back(rtb);
        } else {
            use_zero_copy = false;
            if (rtb) ggml_backend_buffer_free(rtb); if (nb) ggml_backend_buffer_free(nb);
            if (ib) ggml_backend_buffer_free(ib); if (rb) ggml_backend_buffer_free(rb);
            residual_binding = create_standard_binding(context.value, residual_desc);
            input_binding = can_map_standard_view(input_desc) ?
                create_standard_binding(context.value, input_desc) :
                create_packed_standard_binding(context.value, input_desc, packed_input);
            normed_binding = create_standard_binding(context.value, normed_out_desc);
            router_binding = create_standard_binding(context.value, router_out_desc);
        }
    } else {
        residual_binding = create_standard_binding(context.value, residual_desc);
        input_binding = can_map_standard_view(input_desc) ?
            create_standard_binding(context.value, input_desc) :
            create_packed_standard_binding(context.value, input_desc, packed_input);
        normed_binding = create_standard_binding(context.value, normed_out_desc);
        router_binding = create_standard_binding(context.value, router_out_desc);
    }

    ggml_context* ctx = context.value;
    ggml_type out_type = static_cast<ggml_type>(out_proj_quant.ggml_type);
    ggml_tensor* out_w = ggml_new_tensor_2d(ctx, out_type, out_proj_quant.ne0, out_proj_quant.ne1);
    TensorBinding out_w_bind = { out_w, out_w, static_cast<std::size_t>(out_proj_quant.raw_bytes) };
    ggml_tensor* norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, norm_count);
    ggml_type rt_type = static_cast<ggml_type>(router_quant.ggml_type);
    ggml_tensor* router_w = ggml_new_tensor_2d(ctx, rt_type, router_quant.ne0, router_quant.ne1);
    TensorBinding router_w_bind = { router_w, router_w, static_cast<std::size_t>(router_quant.raw_bytes) };

    // Pre-bind cacheable weights.
    bool out_b, out_u, norm_b, norm_u, rt_b, rt_u;
    auto bind = [](ggml_tensor* t, const QuantizedWeightDesc& q, bool& bound, bool& upl) {
        bound = false; upl = false;
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        if (!dev || q.raw_bytes < 4096) return;
        ggml_backend_buffer_t buf = nullptr; void* addr = nullptr;
        if (try_get_cacheable_tensor_buffer(g_backend, dev, t, q.data, (std::size_t)q.raw_bytes, buf, addr, upl))
        { ggml_status st = ggml_backend_tensor_alloc(buf, t, addr); bound = (st == GGML_STATUS_SUCCESS); if (!bound) invalidate_cached_buffer(q.data); }
    };
    bind(out_w, out_proj_quant, out_b, out_u);
    bind(router_w, router_quant, rt_b, rt_u);
    { norm_b = false; norm_u = false;
      std::size_t nb2 = (std::size_t)norm_count * sizeof(float);
      ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
      if (dev && nb2 >= 4096) {
          ggml_backend_buffer_t buf = nullptr; void* addr = nullptr;
          if (try_get_cacheable_tensor_buffer(g_backend, dev, norm_w, norm_weight_data, nb2, buf, addr, norm_u))
          { ggml_status st = ggml_backend_tensor_alloc(buf, norm_w, addr); norm_b = (st == GGML_STATUS_SUCCESS); if (!norm_b) invalidate_cached_buffer(norm_weight_data); }
      }
    }

    // Build graph.
    ggml_tensor* cont_input = ggml_cont(ctx, input_binding.tensor);
    ggml_tensor* cont_res = ggml_cont(ctx, residual_binding.tensor);

    // Phase 1: output projection + residual
    ggml_tensor* inp_2d = (rows == 1) ? ggml_reshape_2d(ctx, cont_input, input_desc.dim1, 1) : cont_input;
    ggml_tensor* out_mm = ggml_mul_mat(ctx, out_w, inp_2d);
    ggml_tensor* out_flat = ggml_reshape_1d(ctx, out_mm, (int64_t)rows * hidden);
    ggml_tensor* res_flat = ggml_reshape_1d(ctx, cont_res, (int64_t)rows * hidden);
    ggml_tensor* res_updated = ggml_add(ctx, res_flat, out_flat);
    ggml_tensor* res_2d = ggml_reshape_2d(ctx, res_updated, hidden, rows);

    // Phase 2: RMSNorm + scale
    ggml_tensor* normed = ggml_rms_norm(ctx, res_2d, eps);
    ggml_tensor* scaled = ggml_mul(ctx, normed, norm_w);

    // Phase 3: Router projection
    ggml_tensor* scaled_2d = (rows == 1) ? ggml_reshape_2d(ctx, scaled, hidden, 1) : scaled;
    ggml_tensor* router_logits = ggml_mul_mat(ctx, router_w, scaled_2d);

    // Outputs: residual, normed, router_logits
    ggml_tensor* out_res = ggml_cpy(ctx, res_updated, residual_binding.tensor);
    ggml_tensor* out_normed = ggml_cpy(ctx, scaled, normed_binding.tensor);
    ggml_tensor* out_router = ggml_cpy(ctx, router_logits, router_binding.tensor);

    ggml_set_output(out_res); ggml_set_output(out_normed); ggml_set_output(out_router);

    ggml_cgraph* graph = ggml_new_graph(ctx);
    if (!graph) { set_last_error("fused_outproj_norm_router: graph failed."); return 0; }
    ggml_build_forward_expand(graph, out_res);
    ggml_build_forward_expand(graph, out_normed);
    ggml_build_forward_expand(graph, out_router);

    BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
    if (!buffer.value) { set_last_error("fused_outproj_norm_router: alloc failed."); return 0; }

    if (!use_zero_copy) {
        upload_binding(residual_binding, residual_desc.data, residual_binding.raw_bytes);
        if (packed_input.empty()) upload_binding(input_binding, input_desc.data, input_binding.raw_bytes);
        else upload_binding(input_binding, packed_input.data(), input_binding.raw_bytes);
    }
    if (!out_b || out_u) upload_binding(out_w_bind, out_proj_quant.data, out_w_bind.raw_bytes);
    if (!rt_b || rt_u) upload_binding(router_w_bind, router_quant.data, router_w_bind.raw_bytes);
    if (!norm_b || norm_u) { TensorBinding tmp = { norm_w, norm_w, (std::size_t)norm_count * sizeof(float) };
        upload_binding(tmp, norm_weight_data, tmp.raw_bytes); }

    ggml_status status = ggml_backend_graph_compute(g_backend, graph);
    if (status != GGML_STATUS_SUCCESS) { set_last_error("fused_outproj_norm_router: compute failed."); return 0; }
    ggml_backend_synchronize(g_backend);

    if (!use_zero_copy) {
        ggml_backend_tensor_get(residual_binding.storage, residual_desc.data, 0, residual_binding.raw_bytes);
        ggml_backend_tensor_get(normed_binding.storage, normed_out_desc.data, 0, normed_binding.raw_bytes);
        ggml_backend_tensor_get(router_binding.storage, router_out_desc.data, 0, router_binding.raw_bytes);
    }

    clear_last_error();
    return 1;
}

} // namespace

TSG_EXPORT int TSGgml_FusedRmsNormMatMulQuantF32(
    TensorView2DDesc result,
    TensorView2DDesc input,
    void* norm_weight_data,
    int norm_weight_count,
    float eps,
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
        return fused_rms_norm_matmul_quant_f32_impl(result, input, norm_weight_data, norm_weight_count, eps, m2_quant);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown error in fused rms_norm_matmul.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_FusedMatMulQuantAddF32(
    TensorView2DDesc residual,
    TensorView2DDesc input,
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
        return fused_matmul_quant_add_f32_impl(residual, input, m2_quant);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown error in fused matmul_add.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_FusedFFNSwiGLUQuantF32(
    TensorView2DDesc residual,
    TensorView2DDesc input,
    void* norm_weight_data,
    int norm_weight_count,
    float eps,
    void* gate_up_data,
    int gate_up_ggml_type,
    std::int64_t gate_up_ne0,
    std::int64_t gate_up_ne1,
    std::int64_t gate_up_raw_bytes,
    void* down_data,
    int down_ggml_type,
    std::int64_t down_ne0,
    std::int64_t down_ne1,
    std::int64_t down_raw_bytes,
    int half_dim)
{
    try
    {
        QuantizedWeightDesc gate_up_quant;
        gate_up_quant.data = gate_up_data;
        gate_up_quant.ggml_type = gate_up_ggml_type;
        gate_up_quant.ne0 = gate_up_ne0;
        gate_up_quant.ne1 = gate_up_ne1;
        gate_up_quant.raw_bytes = gate_up_raw_bytes;

        QuantizedWeightDesc down_quant;
        down_quant.data = down_data;
        down_quant.ggml_type = down_ggml_type;
        down_quant.ne0 = down_ne0;
        down_quant.ne1 = down_ne1;
        down_quant.raw_bytes = down_raw_bytes;

        return fused_ffn_swiglu_quant_f32_impl(
            residual, input, norm_weight_data, norm_weight_count, eps,
            gate_up_quant, down_quant, half_dim);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown error in fused FFN swiGLU.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_FusedVisionMLPF32(
    TensorView2DDesc hidden,
    const float* ln_w, const float* ln_b, int ln_dim, float eps,
    const float* up_w, int up_ne0, int up_ne1, std::int64_t up_bytes,
    const float* up_b, int up_b_dim,
    const float* down_w, int down_ne0, int down_ne1, std::int64_t down_bytes,
    const float* down_b, int down_b_dim)
{
    try
    {
        return fused_vision_mlp_f32_impl(
            hidden, ln_w, ln_b, ln_dim, eps,
            up_w, up_ne0, up_ne1, static_cast<std::size_t>(up_bytes),
            up_b, up_b_dim,
            down_w, down_ne0, down_ne1, static_cast<std::size_t>(down_bytes),
            down_b, down_b_dim);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown error in fused vision MLP.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_FusedVisionAttentionF32(
    TensorView2DDesc hidden,
    const float* ln_w, const float* ln_b, int ln_dim, float eps,
    const float* qkv_w, int qkv_ne0, int qkv_ne1, std::int64_t qkv_bytes,
    const float* qkv_b, int qkv_b_dim,
    const float* out_w, int out_ne0, int out_ne1, std::int64_t out_bytes,
    const float* out_b, int out_b_dim,
    const float* cos_table, const float* sin_table,
    int num_patches, int num_heads, int head_dim, int half_dim,
    float attn_scale)
{
    try { return fused_vision_attention_f32_impl(hidden, ln_w, ln_b, ln_dim, eps,
        qkv_w, qkv_ne0, qkv_ne1, static_cast<std::size_t>(qkv_bytes), qkv_b, qkv_b_dim,
        out_w, out_ne0, out_ne1, static_cast<std::size_t>(out_bytes), out_b, out_b_dim,
        cos_table, sin_table, num_patches, num_heads, head_dim, half_dim, attn_scale); }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in fused vision attention."); return 0; }
}

TSG_EXPORT int TSGgml_FusedOutProjFFNQuantF32(
    TensorView2DDesc residual, TensorView2DDesc input,
    void* out_proj_data, int out_proj_type, std::int64_t out_ne0, std::int64_t out_ne1, std::int64_t out_raw_bytes,
    void* ffn_norm_data, int ffn_norm_count, float eps,
    void* gu_data, int gu_type, std::int64_t gu_ne0, std::int64_t gu_ne1, std::int64_t gu_raw_bytes,
    void* dn_data, int dn_type, std::int64_t dn_ne0, std::int64_t dn_ne1, std::int64_t dn_raw_bytes,
    int half_dim)
{
    try {
        QuantizedWeightDesc out_q = { out_proj_data, out_proj_type, out_ne0, out_ne1, out_raw_bytes };
        QuantizedWeightDesc gu_q  = { gu_data, gu_type, gu_ne0, gu_ne1, gu_raw_bytes };
        QuantizedWeightDesc dn_q  = { dn_data, dn_type, dn_ne0, dn_ne1, dn_raw_bytes };
        return fused_outproj_ffn_quant_f32_impl(residual, input, out_q, ffn_norm_data, ffn_norm_count, eps, gu_q, dn_q, half_dim);
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in fused outproj+FFN."); return 0; }
}

TSG_EXPORT int TSGgml_FusedOutProjNormRouterQuantF32(
    TensorView2DDesc residual, TensorView2DDesc input,
    void* outProjData, int outProjType, std::int64_t outNe0, std::int64_t outNe1, std::int64_t outBytes,
    void* normData, int normCount, float eps,
    TensorView2DDesc normedOut,
    void* routerData, int routerType, std::int64_t routerNe0, std::int64_t routerNe1, std::int64_t routerBytes,
    TensorView2DDesc routerOut)
{
    try {
        QuantizedWeightDesc out_q = { outProjData, outProjType, outNe0, outNe1, outBytes };
        QuantizedWeightDesc rt_q = { routerData, routerType, routerNe0, routerNe1, routerBytes };
        return fused_outproj_norm_router_quant_f32_impl(residual, input, out_q, normData, normCount, eps, normedOut, rt_q, routerOut);
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in fused outproj+norm+router."); return 0; }
}

// Batched MoE expert forward: processes all selected experts in a single GGML graph.
// For each expert: up_proj -> relu_squared -> down_proj -> scale(route_weight) -> accumulate.
// This reduces N*2 GPU dispatches to 1 per MoE layer.
TSG_EXPORT int TSGgml_MoEExpertsForwardF32(
    TensorView2DDesc result,      // [1, outDim] - accumulated output
    TensorView2DDesc input,       // [1, inDim]
    int num_experts,
    void** up_data_ptrs,          // [num_experts] pointers to up weight data
    void** down_data_ptrs,        // [num_experts] pointers to down weight data
    int up_ggml_type,
    std::int64_t up_ne0,          // up weight: ne0 = inDim
    std::int64_t up_ne1,          // up weight: ne1 = intermDim
    std::int64_t up_raw_bytes_each,
    int down_ggml_type,
    std::int64_t down_ne0,        // down weight: ne0 = intermDim
    std::int64_t down_ne1,        // down weight: ne1 = outDim
    std::int64_t down_raw_bytes_each,
    float* route_weights)         // [num_experts]
{
    try
    {
        if (!ensure_backend())
            return 0;

        if (!validate_desc(result, "result") || !validate_desc(input, "input"))
            return 0;

        if (num_experts <= 0 || num_experts > 16)
        {
            set_last_error("MoE: num_experts must be 1..16");
            return 0;
        }

        ggml_type up_qtype = static_cast<ggml_type>(up_ggml_type);
        ggml_type down_qtype = static_cast<ggml_type>(down_ggml_type);

        const std::size_t ctx_size = 4 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("MoE: context init failed");
            return 0;
        }

        // Input / result bindings (zero-copy or standard)
        std::vector<BufferHandle> host_bufs;
        bool zc = can_map_standard_view(input);
        TensorBinding res_bind, inp_bind;

        if (zc)
        {
            ggml_backend_buffer_t rb = nullptr, ib = nullptr;
            bool rok = create_binding_from_host_ptr_2d(context.value, g_backend, result, res_bind, rb);
            bool iok = rok && create_binding_from_host_ptr_2d(context.value, g_backend, input, inp_bind, ib);
            if (rok && iok)
            {
                host_bufs.emplace_back(rb);
                host_bufs.emplace_back(ib);
            }
            else
            {
                if (ib) ggml_backend_buffer_free(ib);
                if (rb) ggml_backend_buffer_free(rb);
                zc = false;
                res_bind = create_standard_binding(context.value, result);
                inp_bind = create_standard_binding(context.value, input);
            }
        }
        else
        {
            res_bind = create_standard_binding(context.value, result);
            inp_bind = create_standard_binding(context.value, input);
        }

        // Weight tensors with caching
        struct WBind { ggml_tensor* t; std::size_t bytes; bool cached; bool needs_upload; };
        std::vector<WBind> up_w(num_experts), dn_w(num_experts);
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);

        for (int e = 0; e < num_experts; e++)
        {
            up_w[e].t = ggml_new_tensor_2d(context.value, up_qtype, up_ne0, up_ne1);
            up_w[e].bytes = static_cast<std::size_t>(up_raw_bytes_each);
            up_w[e].cached = false;
            up_w[e].needs_upload = false;
            if (dev && up_raw_bytes_each >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, up_w[e].t,
                        up_data_ptrs[e], up_w[e].bytes, buf, addr, up_w[e].needs_upload))
                {
                    if (ggml_backend_tensor_alloc(buf, up_w[e].t, addr) == GGML_STATUS_SUCCESS)
                        up_w[e].cached = true;
                    else
                        invalidate_cached_buffer(up_data_ptrs[e]);
                }
            }

            dn_w[e].t = ggml_new_tensor_2d(context.value, down_qtype, down_ne0, down_ne1);
            dn_w[e].bytes = static_cast<std::size_t>(down_raw_bytes_each);
            dn_w[e].cached = false;
            dn_w[e].needs_upload = false;
            if (dev && down_raw_bytes_each >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, dn_w[e].t,
                        down_data_ptrs[e], dn_w[e].bytes, buf, addr, dn_w[e].needs_upload))
                {
                    if (ggml_backend_tensor_alloc(buf, dn_w[e].t, addr) == GGML_STATUS_SUCCESS)
                        dn_w[e].cached = true;
                    else
                        invalidate_cached_buffer(down_data_ptrs[e]);
                }
            }
        }

        // Build computation graph: for each expert, up -> relu -> sqr -> down -> scale -> accumulate
        ggml_tensor* accum = nullptr;
        for (int e = 0; e < num_experts; e++)
        {
            ggml_tensor* up_out = ggml_mul_mat(context.value, up_w[e].t, inp_bind.tensor);
            ggml_tensor* relu_out = ggml_relu(context.value, up_out);
            ggml_tensor* sq_out = ggml_sqr(context.value, relu_out);
            ggml_tensor* dn_out = ggml_mul_mat(context.value, dn_w[e].t, sq_out);
            ggml_tensor* scaled = ggml_scale(context.value, dn_out, route_weights[e]);
            accum = (accum == nullptr) ? scaled : ggml_add(context.value, accum, scaled);
        }

        ggml_tensor* out = ggml_cpy(context.value, accum, res_bind.tensor);
        ggml_set_output(out);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        ggml_build_forward_expand(graph, out);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (!buffer.value)
        {
            set_last_error("MoE: buffer alloc failed");
            return 0;
        }

        // Upload input (if not zero-copy)
        if (!zc)
            upload_binding(inp_bind, input.data, inp_bind.raw_bytes);

        // Upload weight data
        for (int e = 0; e < num_experts; e++)
        {
            if (!up_w[e].cached || up_w[e].needs_upload)
                ggml_backend_tensor_set(up_w[e].t, up_data_ptrs[e], 0, up_w[e].bytes);
            if (!dn_w[e].cached || dn_w[e].needs_upload)
                ggml_backend_tensor_set(dn_w[e].t, down_data_ptrs[e], 0, dn_w[e].bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("MoE: graph compute failed");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!zc)
            ggml_backend_tensor_get(res_bind.storage, result.data, 0, res_bind.raw_bytes);

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
        set_last_error("Unknown MoE experts forward failure.");
        return 0;
    }
}

// ============================================================================
// Batched MoE SwiGLU expert forward: SwiGLU activation pattern (qwen3 / mixtral).
// For each selected expert e:
//   gate_e = input @ gate_w[e]        // [interm_dim]
//   up_e   = input @ up_w[e]          // [interm_dim]
//   inner  = silu(gate_e) * up_e      // [interm_dim]
//   out_e  = inner @ down_w[e]        // [out_dim]
// result = sum_e route_w[e] * out_e   // [out_dim]
// All expert ops are batched into a single GGML graph, reducing
// 4 * num_experts GPU dispatches to a single graph submission.
// ============================================================================
TSG_EXPORT int TSGgml_MoEExpertsSwiGLUForwardF32(
    TensorView2DDesc result,            // [1, out_dim] (accumulated output)
    TensorView2DDesc input,             // [1, in_dim]
    int num_experts,
    void** gate_data_ptrs,              // [num_experts] gate weight data
    void** up_data_ptrs,                // [num_experts] up weight data
    void** down_data_ptrs,              // [num_experts] down weight data
    int gate_ggml_type,
    std::int64_t gate_ne0,              // = in_dim
    std::int64_t gate_ne1,              // = interm_dim
    std::int64_t gate_raw_bytes_each,
    int up_ggml_type,
    std::int64_t up_ne0,                // = in_dim
    std::int64_t up_ne1,                // = interm_dim
    std::int64_t up_raw_bytes_each,
    int down_ggml_type,
    std::int64_t down_ne0,              // = interm_dim
    std::int64_t down_ne1,              // = out_dim
    std::int64_t down_raw_bytes_each,
    float* route_weights)               // [num_experts]
{
    try
    {
        if (!ensure_backend())
            return 0;

        if (!validate_desc(result, "result") || !validate_desc(input, "input"))
            return 0;

        if (num_experts <= 0 || num_experts > 32)
        {
            set_last_error("MoE SwiGLU: num_experts must be 1..32");
            return 0;
        }

        ggml_type gate_qtype = static_cast<ggml_type>(gate_ggml_type);
        ggml_type up_qtype   = static_cast<ggml_type>(up_ggml_type);
        ggml_type down_qtype = static_cast<ggml_type>(down_ggml_type);

        const std::size_t ctx_size = 8 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("MoE SwiGLU: context init failed");
            return 0;
        }

        // Input / result bindings (zero-copy when possible)
        std::vector<BufferHandle> host_bufs;
        bool zc = can_map_standard_view(input);
        TensorBinding res_bind, inp_bind;

        if (zc)
        {
            ggml_backend_buffer_t rb = nullptr, ib = nullptr;
            bool rok = create_binding_from_host_ptr_2d(context.value, g_backend, result, res_bind, rb);
            bool iok = rok && create_binding_from_host_ptr_2d(context.value, g_backend, input, inp_bind, ib);
            if (rok && iok)
            {
                host_bufs.emplace_back(rb);
                host_bufs.emplace_back(ib);
            }
            else
            {
                if (ib) ggml_backend_buffer_free(ib);
                if (rb) ggml_backend_buffer_free(rb);
                zc = false;
                res_bind = create_standard_binding(context.value, result);
                inp_bind = create_standard_binding(context.value, input);
            }
        }
        else
        {
            res_bind = create_standard_binding(context.value, result);
            inp_bind = create_standard_binding(context.value, input);
        }

        // Per-expert weight tensors with weight cache attempts.
        struct WBind { ggml_tensor* t; std::size_t bytes; bool cached; bool needs_upload; };
        std::vector<WBind> gate_w(num_experts), up_w(num_experts), dn_w(num_experts);
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);

        auto bind_quant = [&](WBind& wb, ggml_type qtype, std::int64_t ne0, std::int64_t ne1,
                              std::int64_t raw_bytes, void* data) {
            wb.t = ggml_new_tensor_2d(context.value, qtype, ne0, ne1);
            wb.bytes = static_cast<std::size_t>(raw_bytes);
            wb.cached = false;
            wb.needs_upload = false;
            if (dev && raw_bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, wb.t,
                        data, wb.bytes, buf, addr, wb.needs_upload))
                {
                    if (ggml_backend_tensor_alloc(buf, wb.t, addr) == GGML_STATUS_SUCCESS)
                        wb.cached = true;
                    else
                        invalidate_cached_buffer(data);
                }
            }
        };

        for (int e = 0; e < num_experts; e++)
        {
            bind_quant(gate_w[e], gate_qtype, gate_ne0, gate_ne1, gate_raw_bytes_each, gate_data_ptrs[e]);
            bind_quant(up_w[e],   up_qtype,   up_ne0,   up_ne1,   up_raw_bytes_each,   up_data_ptrs[e]);
            bind_quant(dn_w[e],   down_qtype, down_ne0, down_ne1, down_raw_bytes_each, down_data_ptrs[e]);
        }

        // Build computation graph: for each expert,
        //   (silu(input @ gate_w) * (input @ up_w)) @ down_w * route_w  --> accum
        ggml_tensor* accum = nullptr;
        for (int e = 0; e < num_experts; e++)
        {
            ggml_tensor* gate_out = ggml_mul_mat(context.value, gate_w[e].t, inp_bind.tensor);
            ggml_tensor* up_out   = ggml_mul_mat(context.value, up_w[e].t,   inp_bind.tensor);
            ggml_tensor* silu_out = ggml_silu(context.value, gate_out);
            ggml_tensor* prod     = ggml_mul(context.value, silu_out, up_out);
            ggml_tensor* dn_out   = ggml_mul_mat(context.value, dn_w[e].t, prod);
            ggml_tensor* scaled   = ggml_scale(context.value, dn_out, route_weights[e]);
            accum = (accum == nullptr) ? scaled : ggml_add(context.value, accum, scaled);
        }

        ggml_tensor* out = ggml_cpy(context.value, accum, res_bind.tensor);
        ggml_set_output(out);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        ggml_build_forward_expand(graph, out);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (!buffer.value)
        {
            set_last_error("MoE SwiGLU: buffer alloc failed");
            return 0;
        }

        if (!zc)
            upload_binding(inp_bind, input.data, inp_bind.raw_bytes);

        for (int e = 0; e < num_experts; e++)
        {
            if (!gate_w[e].cached || gate_w[e].needs_upload)
                ggml_backend_tensor_set(gate_w[e].t, gate_data_ptrs[e], 0, gate_w[e].bytes);
            if (!up_w[e].cached || up_w[e].needs_upload)
                ggml_backend_tensor_set(up_w[e].t, up_data_ptrs[e], 0, up_w[e].bytes);
            if (!dn_w[e].cached || dn_w[e].needs_upload)
                ggml_backend_tensor_set(dn_w[e].t, down_data_ptrs[e], 0, dn_w[e].bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("MoE SwiGLU: graph compute failed");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!zc)
            ggml_backend_tensor_get(res_bind.storage, result.data, 0, res_bind.raw_bytes);

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
        set_last_error("Unknown MoE SwiGLU experts forward failure.");
        return 0;
    }
}

// ============================================================================
// Extended MoE SwiGLU forward: routed experts + optional shared expert + fused
// residual add, all in a single GGML graph submission.
//
//   residual += sum_e route_w[e] * down_w[e] * (silu(input @ gate_w[e]) * (input @ up_w[e]))
//             + (use_shared ? shared_scalar * shared_down @ (silu(input @ shared_gate)
//                                                            * (input @ shared_up)) : 0)
//
// This eliminates ~37 dispatches per MoE layer per token (8*4 routed + 4 shared + 1 add)
// down to 1.
// ============================================================================
TSG_EXPORT int TSGgml_MoEExpertsSwiGLUResidualF32(
    TensorView2DDesc residual,          // [1, hidden_size] - in/out, accumulated into
    TensorView2DDesc input,             // [1, hidden_size] - normalized input for MoE
    int num_experts,
    void** gate_data_ptrs,              // [num_experts] gate weight data
    void** up_data_ptrs,                // [num_experts] up weight data
    void** down_data_ptrs,              // [num_experts] down weight data
    int gate_ggml_type,
    std::int64_t gate_ne0,
    std::int64_t gate_ne1,
    std::int64_t gate_raw_bytes_each,
    int up_ggml_type,
    std::int64_t up_ne0,
    std::int64_t up_ne1,
    std::int64_t up_raw_bytes_each,
    int down_ggml_type,
    std::int64_t down_ne0,
    std::int64_t down_ne1,
    std::int64_t down_raw_bytes_each,
    float* route_weights,               // [num_experts]
    int use_shared,                     // 0 / 1
    void* shared_gate_data,
    void* shared_up_data,
    void* shared_down_data,
    int shared_gate_ggml_type,
    std::int64_t shared_gate_ne0,
    std::int64_t shared_gate_ne1,
    std::int64_t shared_gate_raw_bytes,
    int shared_up_ggml_type,
    std::int64_t shared_up_ne0,
    std::int64_t shared_up_ne1,
    std::int64_t shared_up_raw_bytes,
    int shared_down_ggml_type,
    std::int64_t shared_down_ne0,
    std::int64_t shared_down_ne1,
    std::int64_t shared_down_raw_bytes,
    float shared_scalar)
{
    try
    {
        if (!ensure_backend())
            return 0;

        if (!validate_desc(residual, "residual") || !validate_desc(input, "input"))
            return 0;

        if (num_experts <= 0 || num_experts > 32)
        {
            set_last_error("MoE SwiGLU residual: num_experts must be 1..32");
            return 0;
        }

        ggml_type gate_qtype = static_cast<ggml_type>(gate_ggml_type);
        ggml_type up_qtype   = static_cast<ggml_type>(up_ggml_type);
        ggml_type down_qtype = static_cast<ggml_type>(down_ggml_type);

        const std::size_t ctx_size = 12 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("MoE SwiGLU residual: context init failed");
            return 0;
        }

        std::vector<BufferHandle> host_bufs;
        bool zc = can_map_standard_view(input) && can_map_standard_view(residual);
        TensorBinding res_bind, inp_bind;

        if (zc)
        {
            ggml_backend_buffer_t rb = nullptr, ib = nullptr;
            bool rok = create_binding_from_host_ptr_2d(context.value, g_backend, residual, res_bind, rb);
            bool iok = rok && create_binding_from_host_ptr_2d(context.value, g_backend, input, inp_bind, ib);
            if (rok && iok)
            {
                host_bufs.emplace_back(rb);
                host_bufs.emplace_back(ib);
            }
            else
            {
                if (ib) ggml_backend_buffer_free(ib);
                if (rb) ggml_backend_buffer_free(rb);
                zc = false;
                res_bind = create_standard_binding(context.value, residual);
                inp_bind = create_standard_binding(context.value, input);
            }
        }
        else
        {
            res_bind = create_standard_binding(context.value, residual);
            inp_bind = create_standard_binding(context.value, input);
        }

        struct WBind { ggml_tensor* t; std::size_t bytes; bool cached; bool needs_upload; };
        std::vector<WBind> gate_w(num_experts), up_w(num_experts), dn_w(num_experts);
        WBind sh_g{}, sh_u{}, sh_d{};
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);

        auto bind_quant = [&](WBind& wb, ggml_type qtype, std::int64_t ne0, std::int64_t ne1,
                              std::int64_t raw_bytes, void* data) {
            wb.t = ggml_new_tensor_2d(context.value, qtype, ne0, ne1);
            wb.bytes = static_cast<std::size_t>(raw_bytes);
            wb.cached = false;
            wb.needs_upload = false;
            if (dev && raw_bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, wb.t,
                        data, wb.bytes, buf, addr, wb.needs_upload))
                {
                    if (ggml_backend_tensor_alloc(buf, wb.t, addr) == GGML_STATUS_SUCCESS)
                        wb.cached = true;
                    else
                        invalidate_cached_buffer(data);
                }
            }
        };

        for (int e = 0; e < num_experts; e++)
        {
            bind_quant(gate_w[e], gate_qtype, gate_ne0, gate_ne1, gate_raw_bytes_each, gate_data_ptrs[e]);
            bind_quant(up_w[e],   up_qtype,   up_ne0,   up_ne1,   up_raw_bytes_each,   up_data_ptrs[e]);
            bind_quant(dn_w[e],   down_qtype, down_ne0, down_ne1, down_raw_bytes_each, down_data_ptrs[e]);
        }

        bool has_shared = (use_shared != 0)
                       && shared_gate_data != nullptr
                       && shared_up_data != nullptr
                       && shared_down_data != nullptr;

        if (has_shared)
        {
            bind_quant(sh_g, static_cast<ggml_type>(shared_gate_ggml_type),
                shared_gate_ne0, shared_gate_ne1, shared_gate_raw_bytes, shared_gate_data);
            bind_quant(sh_u, static_cast<ggml_type>(shared_up_ggml_type),
                shared_up_ne0, shared_up_ne1, shared_up_raw_bytes, shared_up_data);
            bind_quant(sh_d, static_cast<ggml_type>(shared_down_ggml_type),
                shared_down_ne0, shared_down_ne1, shared_down_raw_bytes, shared_down_data);
        }

        // Build computation graph: routed experts (silu(gate)*up @ down * route_w) accumulated,
        // plus optional shared expert (silu(gate)*up @ down * scalar) accumulated, plus residual.
        ggml_tensor* accum = nullptr;
        for (int e = 0; e < num_experts; e++)
        {
            ggml_tensor* gate_out = ggml_mul_mat(context.value, gate_w[e].t, inp_bind.tensor);
            ggml_tensor* up_out   = ggml_mul_mat(context.value, up_w[e].t,   inp_bind.tensor);
            ggml_tensor* silu_out = ggml_silu(context.value, gate_out);
            ggml_tensor* prod     = ggml_mul(context.value, silu_out, up_out);
            ggml_tensor* dn_out   = ggml_mul_mat(context.value, dn_w[e].t, prod);
            ggml_tensor* scaled   = ggml_scale(context.value, dn_out, route_weights[e]);
            accum = (accum == nullptr) ? scaled : ggml_add(context.value, accum, scaled);
        }

        if (has_shared)
        {
            ggml_tensor* sg_out = ggml_mul_mat(context.value, sh_g.t, inp_bind.tensor);
            ggml_tensor* su_out = ggml_mul_mat(context.value, sh_u.t, inp_bind.tensor);
            ggml_tensor* ssilu  = ggml_silu(context.value, sg_out);
            ggml_tensor* sprod  = ggml_mul(context.value, ssilu, su_out);
            ggml_tensor* sdn    = ggml_mul_mat(context.value, sh_d.t, sprod);
            ggml_tensor* sscaled = ggml_scale(context.value, sdn, shared_scalar);
            accum = (accum == nullptr) ? sscaled : ggml_add(context.value, accum, sscaled);
        }

        // residual += accum (in place)
        ggml_tensor* sum = ggml_add(context.value, res_bind.tensor, accum);
        ggml_tensor* out = ggml_cpy(context.value, sum, res_bind.tensor);
        ggml_set_output(out);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        ggml_build_forward_expand(graph, out);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (!buffer.value)
        {
            set_last_error("MoE SwiGLU residual: buffer alloc failed");
            return 0;
        }

        if (!zc)
        {
            upload_binding(inp_bind, input.data, inp_bind.raw_bytes);
            upload_binding(res_bind, residual.data, res_bind.raw_bytes);
        }

        for (int e = 0; e < num_experts; e++)
        {
            if (!gate_w[e].cached || gate_w[e].needs_upload)
                ggml_backend_tensor_set(gate_w[e].t, gate_data_ptrs[e], 0, gate_w[e].bytes);
            if (!up_w[e].cached || up_w[e].needs_upload)
                ggml_backend_tensor_set(up_w[e].t, up_data_ptrs[e], 0, up_w[e].bytes);
            if (!dn_w[e].cached || dn_w[e].needs_upload)
                ggml_backend_tensor_set(dn_w[e].t, down_data_ptrs[e], 0, dn_w[e].bytes);
        }

        if (has_shared)
        {
            if (!sh_g.cached || sh_g.needs_upload)
                ggml_backend_tensor_set(sh_g.t, shared_gate_data, 0, sh_g.bytes);
            if (!sh_u.cached || sh_u.needs_upload)
                ggml_backend_tensor_set(sh_u.t, shared_up_data, 0, sh_u.bytes);
            if (!sh_d.cached || sh_d.needs_upload)
                ggml_backend_tensor_set(sh_d.t, shared_down_data, 0, sh_d.bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("MoE SwiGLU residual: graph compute failed");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!zc)
            ggml_backend_tensor_get(res_bind.storage, residual.data, 0, res_bind.raw_bytes);

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
        set_last_error("Unknown MoE SwiGLU residual forward failure.");
        return 0;
    }
}
