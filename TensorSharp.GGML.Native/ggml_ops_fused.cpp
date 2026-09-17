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

    ggml_status status = tsg::compute_graph(g_backend, graph);
    if (status != GGML_STATUS_SUCCESS)
    {
        set_last_error("Graph execution failed for fused rms_norm_matmul.");
        return 0;
    }
    finalize_compute(use_zero_copy, result_binding.storage, result_desc.data, result_binding.raw_bytes);

    clear_last_error();
    return 1;
}

// Per-rank parking for a tensor-parallel FFN graph: the driver runs it after
// this function returns, so the pooled context and the backend buffer have to
// outlive the call. Recycled when that rank builds its next one.
namespace
{
    struct FfnTpPending
    {
        PooledContextHandle context;
        BufferHandle buffer{ nullptr };
        std::vector<BufferHandle> host_ptr_buffers;
        std::vector<float> packed_input;
        tsg::TpRankPlan plan;

        void reset()
        {
            plan.clear();
            host_ptr_buffers.clear();
            packed_input.clear();
            buffer = BufferHandle(nullptr);
            context = PooledContextHandle();
        }
    };
    FfnTpPending g_ffn_tp[tsg::TSG_MAX_DEVICES];
    // Same parking for the generic row-parallel matmul + residual add.
    FfnTpPending g_mmadd_tp[tsg::TSG_MAX_DEVICES];
}

// ============================================================================
// Fused Quantized MatMul + Add: single GPU dispatch.
// residual += matmul(input, quant_weight)
// ============================================================================

int fused_matmul_quant_add_f32_impl(
    const TensorView2DDesc& residual_desc,
    const TensorView2DDesc& input_desc,
    const QuantizedWeightDesc& m2_quant,
    int tp_degree,
    void** tp_plan_out)
{
    if (!ensure_backend())
        return 0;

    // Tensor-parallel build mode. This is the generic row-parallel layer: the
    // matmul is a partial sum, the ranks add theirs together, then the residual
    // add closes the block. Cutting at the matmul is what lets an attention or
    // SSM output projection run as one graph per rank instead of a linear, a
    // collective and an add, each a separate host round trip.
    // Plan mode needs BOTH a place to put the plan and a group that actually
        // spans ranks. The degree alone is not enough: one local rank per node is
        // a legitimate distributed shape. Nor is the pointer alone - this entry is
        // marshalled as `out IntPtr`, so C# hands it a non-null slot on EVERY call,
        // including a plain single-GPU decode. Gating on the pointer by itself put
        // that decode into plan mode: the graph was built, never executed, and the
        // caller read an untouched logits buffer - fluent-looking <pad> spam at
        // thousands of tokens/sec because no compute had happened at all.
        const bool tp_mode = tp_plan_out != nullptr &&
            (tp_degree > 1 || tsg::tp_global_degree(tp_degree) > 1);
    if (tp_mode)
    {
        *tp_plan_out = nullptr;
        g_mmadd_tp[tsg::g_active_rank].reset();
    }

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

    if (tp_mode)
    {
        auto& pending = g_mmadd_tp[tsg::g_active_rank];
        pending.plan.clear();
        pending.plan.graph = graph;
        pending.plan.ar_tensor.push_back(mm);
        if (!tp_plan_segments(pending.plan, pending.plan.ar_tensor))
        {
            pending.reset();
            return 0;
        }
        if (!use_zero_copy)
        {
            pending.plan.out_tensor = residual_binding.storage;
            pending.plan.out_host = residual_desc.data;
            pending.plan.out_bytes = residual_binding.raw_bytes;
        }
        pending.context = std::move(context);
        pending.buffer = std::move(buffer);
        pending.host_ptr_buffers = std::move(host_ptr_buffers);
        pending.packed_input = std::move(packed_input);
        *tp_plan_out = &pending.plan;
        clear_last_error();
        return 1;
    }

    ggml_status status = tsg::compute_graph(g_backend, graph);
    if (status != GGML_STATUS_SUCCESS)
    {
        set_last_error("Graph execution failed for fused matmul_add.");
        return 0;
    }
    finalize_compute(use_zero_copy, residual_binding.storage, residual_desc.data, residual_binding.raw_bytes);

    clear_last_error();
    return 1;
}

// Release every rank's parked tensor-parallel matmul+add graph.
TSG_EXPORT void TSGgml_ReleaseFusedMatmulAddTpGraphs()
{
    for (int r = 0; r < tsg::TSG_MAX_DEVICES; ++r)
    {
        if (g_mmadd_tp[r].plan.graph == nullptr && g_mmadd_tp[r].context.value == nullptr)
            continue;
        tsg::ScopedRank rank(r);
        g_mmadd_tp[r].reset();
    }
}

namespace
{
    // Byte ceiling for the activation set of ONE fused FFN graph. The op is
    // row-splittable, so a prompt whose single-shot graph would exceed this is
    // walked in slabs instead of demanding one enormous device buffer -- the
    // 27B/1073-token multimodal prefill that motivated this asked for 645 MiB
    // in a single cudaMalloc and died. Override with TS_GGML_FFN_GRAPH_BUDGET_MB
    // (0 disables slabbing entirely).
    std::size_t ffn_graph_budget_bytes()
    {
        static const std::size_t s_budget = []() -> std::size_t {
            const char* e = std::getenv("TS_GGML_FFN_GRAPH_BUDGET_MB");
            if (e != nullptr && e[0] != '\0')
            {
                char* end = nullptr;
                const long long mb = std::strtoll(e, &end, 10);
                if (end != e && mb >= 0)
                    return static_cast<std::size_t>(mb) * 1024ull * 1024ull;
            }
            return 512ull * 1024ull * 1024ull;
        }();
        return s_budget;
    }

    // Largest row slab whose graph activations fit the budget, clamped to
    // [1, rows]. `bytes_per_row` is the caller's per-row activation footprint.
    int ffn_rows_per_slab(int rows, std::size_t bytes_per_row)
    {
        const std::size_t budget = ffn_graph_budget_bytes();
        if (budget == 0 || bytes_per_row == 0)
            return rows;
        const std::size_t fit = budget / bytes_per_row;
        if (fit == 0)
            return 1;
        return fit >= static_cast<std::size_t>(rows) ? rows : static_cast<int>(fit);
    }
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
static int fused_ffn_swiglu_quant_f32_slab(
    const TensorView2DDesc& residual_desc,
    const TensorView2DDesc& input_desc,
    void* norm_weight_data,
    int norm_weight_count,
    float eps,
    const QuantizedWeightDesc& gate_up_quant,
    const QuantizedWeightDesc& down_quant,
    int half_dim,
    int tp_degree,
    void** tp_plan_out,
    int* alloc_failed_out)
{
    if (alloc_failed_out != nullptr)
        *alloc_failed_out = 0;
    if (!ensure_backend())
        return 0;

    // Tensor-parallel build mode: the down projection is row-parallel, so its
    // output is this rank's partial sum and the residual add has to wait for the
    // ranks to agree. Build the graph, mark that cut, hand it back.
    // Plan mode needs BOTH a place to put the plan and a group that actually
        // spans ranks. The degree alone is not enough: one local rank per node is
        // a legitimate distributed shape. Nor is the pointer alone - this entry is
        // marshalled as `out IntPtr`, so C# hands it a non-null slot on EVERY call,
        // including a plain single-GPU decode. Gating on the pointer by itself put
        // that decode into plan mode: the graph was built, never executed, and the
        // caller read an untouched logits buffer - fluent-looking <pad> spam at
        // thousands of tokens/sec because no compute had happened at all.
        const bool tp_mode = tp_plan_out != nullptr &&
            (tp_degree > 1 || tsg::tp_global_degree(tp_degree) > 1);
    if (tp_mode)
    {
        *tp_plan_out = nullptr;
        g_ffn_tp[tsg::g_active_rank].reset();
    }

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

    // silu(gate) * up in ONE node straight off the strided halves. The GLU op
    // only needs ggml_is_contiguous_1 (rows contiguous, arbitrary row stride),
    // which a [half_dim, rows] view of the packed gate_up output satisfies on
    // every backend (CPU ops.cpp, ggml-cuda/unary.cu, ggml-metal, ggml-vulkan
    // all read src->nb[1]). The previous cont/cont/silu/mul chain materialized
    // FOUR extra [rows, half_dim] F32 tensors -- at 1073 rows x 17408 that is
    // 285 MiB of activations and 3 extra kernel launches per layer.
    ggml_tensor* swiglu = ggml_swiglu_split(context.value, gate_view, up_view);

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
    // The residual base is both a graph input (its cont feeds the add) and the
    // destination of the terminal cpy. Marking it OUTPUT keeps ggml_gallocr from
    // recycling its memory for a later intermediate (ggml-alloc.c
    // ggml_gallocr_free_node skips OUTPUT tensors) and from picking it as an
    // in-place target.
    ggml_set_output(residual_binding.storage);

    ggml_cgraph* graph = ggml_new_graph(context.value);
    if (graph == nullptr)
    {
        set_last_error("fused_ffn_swiglu: failed to create graph.");
        return 0;
    }
    ggml_build_forward_expand(graph, output_tensor);

    // Allocate the activations out of the persistent, liveness-planning gallocr
    // instead of one fresh device buffer per call. ggml_backend_alloc_ctx_tensors
    // has no notion of liveness: it sums EVERY tensor in the context, so a
    // 1073-row FFN asked for 595 MiB in a single cudaMalloc, per layer, per
    // forward -- both the OOM in the multimodal prefill and a per-layer
    // alloc/free that fragments VRAM and stalls the driver. gallocr reuses one
    // grown-once buffer across every call and only holds the live set (~2.4x
    // less here). TP build mode parks the graph for deferred execution, so it
    // keeps its own buffer -- a shared allocator would be re-planned out from
    // under the parked graph.
    BufferHandle buffer(nullptr);
    if (tp_mode || !alloc_graph_reuse_gallocr(graph))
    {
        buffer.value = ggml_backend_alloc_ctx_tensors(context.value, g_backend);
        if (buffer.value == nullptr)
        {
            if (alloc_failed_out != nullptr)
                *alloc_failed_out = 1;
            set_last_error("fused_ffn_swiglu: failed to allocate backend buffer.");
            return 0;
        }
    }

    // Drain the previous call's GPU work before overwriting the shared buffer
    // (no-op on the eager-sync path and on the per-call fallback).
    host_read_barrier();

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

    if (tp_mode)
    {
        auto& pending = g_ffn_tp[tsg::g_active_rank];
        pending.plan.clear();
        pending.plan.graph = graph;
        pending.plan.ar_tensor.push_back(down_mm);
        if (!tp_plan_segments(pending.plan, pending.plan.ar_tensor))
        {
            pending.reset();
            return 0;
        }
        // Zero-copy bindings write the result straight into the caller's host
        // memory; otherwise the driver copies it back after the last segment.
        if (!use_zero_copy)
        {
            pending.plan.out_tensor = residual_binding.storage;
            pending.plan.out_host = residual_desc.data;
            pending.plan.out_bytes = residual_binding.raw_bytes;
        }
        pending.context = std::move(context);
        pending.buffer = std::move(buffer);
        pending.host_ptr_buffers = std::move(host_ptr_buffers);
        pending.packed_input = std::move(packed_input);
        *tp_plan_out = &pending.plan;
        clear_last_error();
        return 1;
    }

    ggml_status status = tsg::compute_graph(g_backend, graph);
    if (status != GGML_STATUS_SUCCESS)
    {
        set_last_error("fused_ffn_swiglu: graph execution failed.");
        return 0;
    }
    finalize_compute(use_zero_copy, residual_binding.storage, residual_desc.data, residual_binding.raw_bytes);
    // Drain the queued async download before the per-call fallback buffer frees
    // (no-op on the reuse-gallocr path, where buffer.value == nullptr).
    if (buffer.value != nullptr) host_read_barrier();

    clear_last_error();
    return 1;
}

// Row-slabbed driver for the fused SwiGLU FFN. Every row is independent, so a
// prompt whose single-shot activation set would exceed the graph budget (or
// whose allocation fails outright on a loaded GPU) is walked in slabs instead of
// failing the turn. TP build mode parks one graph per call and is never slabbed.
int fused_ffn_swiglu_quant_f32_impl(
    const TensorView2DDesc& residual_desc,
    const TensorView2DDesc& input_desc,
    void* norm_weight_data,
    int norm_weight_count,
    float eps,
    const QuantizedWeightDesc& gate_up_quant,
    const QuantizedWeightDesc& down_quant,
    int half_dim,
    int tp_degree,
    void** tp_plan_out)
{
    // Plan mode needs BOTH a place to put the plan and a group that actually
        // spans ranks. The degree alone is not enough: one local rank per node is
        // a legitimate distributed shape. Nor is the pointer alone - this entry is
        // marshalled as `out IntPtr`, so C# hands it a non-null slot on EVERY call,
        // including a plain single-GPU decode. Gating on the pointer by itself put
        // that decode into plan mode: the graph was built, never executed, and the
        // caller read an untouched logits buffer - fluent-looking <pad> spam at
        // thousands of tokens/sec because no compute had happened at all.
        const bool tp_mode = tp_plan_out != nullptr &&
            (tp_degree > 1 || tsg::tp_global_degree(tp_degree) > 1);
    const int rows = input_desc.dim0;
    if (tp_mode || rows <= 1 || half_dim <= 0 || input_desc.dim1 <= 0)
    {
        return fused_ffn_swiglu_quant_f32_slab(residual_desc, input_desc, norm_weight_data,
            norm_weight_count, eps, gate_up_quant, down_quant, half_dim, tp_degree, tp_plan_out, nullptr);
    }

    // Live activation set per row under gallocr: the residual base and its cont
    // stay live across the whole graph, the norm chain reuses one hidden-sized
    // slot, and the peak adds gate_up (2*half) + swiglu (half). Budget against
    // that rather than the un-reused sum so a comfortably-fitting prompt is
    // still done in one dispatch.
    const std::size_t bytes_per_row = sizeof(float) * (
        3ull * static_cast<std::size_t>(input_desc.dim1) +
        3ull * static_cast<std::size_t>(half_dim));

    int slab = ffn_rows_per_slab(rows, bytes_per_row);
    int start = 0;
    while (start < rows)
    {
        const int count = std::min(slab, rows - start);
        int alloc_failed = 0;
        if (fused_ffn_swiglu_quant_f32_slab(
                slice_rows_2d(residual_desc, start, count),
                slice_rows_2d(input_desc, start, count),
                norm_weight_data, norm_weight_count, eps,
                gate_up_quant, down_quant, half_dim, 1, nullptr, &alloc_failed) != 0)
        {
            start += count;
            continue;
        }
        // Only an ALLOCATION failure is retryable: the slab died before its
        // graph ran, so its rows of `residual` are untouched and narrowing is
        // exact. A compute failure may have partially written them, and
        // re-running would add the FFN output twice -- fail the call instead.
        if (alloc_failed == 0 || slab <= 1)
            return 0;
        slab = std::max(1, slab / 2);
    }
    clear_last_error();
    return 1;
}

// Release every rank's parked tensor-parallel FFN graph.
TSG_EXPORT void TSGgml_ReleaseFusedFfnTpGraphs()
{
    for (int r = 0; r < tsg::TSG_MAX_DEVICES; ++r)
    {
        if (g_ffn_tp[r].plan.graph == nullptr && g_ffn_tp[r].context.value == nullptr)
            continue;
        tsg::ScopedRank rank(r);
        g_ffn_tp[r].reset();
    }
}

// ============================================================================
// Fused dense FFN PROJECTION (no residual fold). Writes the FFN output to a
// separate tensor so callers that apply a post-activation norm before the
// residual add (e.g. Gemma 4's post_ffw_norm) can run it on the small output:
//
//   output = down_W^T @ ( act(gate_part) * up_part ),
//     [gate_part | up_part] = gate_up_W^T @ rms_norm(input, normW, eps)
//   act = (act_type == 1) ? gelu(tanh approx) : silu
//
// Like fused_ffn_swiglu_quant_f32_impl this keeps the large [rows, 2*half_dim]
// gate_up intermediate resident on the device (the dominant prefill cost on the
// GGML CUDA backend), but returns the projection instead of folding it into a
// residual. rms_norm uses the same loaded weight as Ops.RMSNorm, so the result
// is identical to the unfused norm+gate_up+act+down chain for any model whose
// norm-weight convention is already baked into the stored weight (Gemma 4
// included).
// ============================================================================
int fused_ffn_act_project_quant_f32_impl(
    const TensorView2DDesc& output_desc,
    const TensorView2DDesc& input_desc,
    void* norm_weight_data,
    int norm_weight_count,
    float eps,
    const QuantizedWeightDesc& gate_up_quant,
    const QuantizedWeightDesc& down_quant,
    int half_dim,
    int act_type)
{
    if (!ensure_backend())
        return 0;

    if (!validate_desc(output_desc, "output") || !validate_desc(input_desc, "input"))
        return 0;

    if (norm_weight_data == nullptr || norm_weight_count <= 0)
    {
        set_last_error("fused_ffn_act: invalid norm weight.");
        return 0;
    }
    if (gate_up_quant.data == nullptr || gate_up_quant.ne0 <= 0 || gate_up_quant.ne1 <= 0 || gate_up_quant.raw_bytes <= 0)
    {
        set_last_error("fused_ffn_act: invalid gate_up weight descriptor.");
        return 0;
    }
    if (down_quant.data == nullptr || down_quant.ne0 <= 0 || down_quant.ne1 <= 0 || down_quant.raw_bytes <= 0)
    {
        set_last_error("fused_ffn_act: invalid down weight descriptor.");
        return 0;
    }

    const int rows = input_desc.dim0;
    const int hidden = input_desc.dim1;
    const int gate_up_out = static_cast<int>(gate_up_quant.ne1);
    const int down_in = static_cast<int>(down_quant.ne0);
    const int down_out = static_cast<int>(down_quant.ne1);

    if (output_desc.dim0 != rows || output_desc.dim1 != hidden)
    {
        set_last_error("fused_ffn_act: output shape mismatch.");
        return 0;
    }
    if (norm_weight_count != hidden)
    {
        set_last_error("fused_ffn_act: norm_weight_count != hidden.");
        return 0;
    }
    if (gate_up_quant.ne0 != hidden)
    {
        set_last_error("fused_ffn_act: gate_up.ne0 != hidden.");
        return 0;
    }
    if (gate_up_out != 2 * half_dim)
    {
        set_last_error("fused_ffn_act: gate_up.ne1 != 2*half_dim.");
        return 0;
    }
    if (down_in != half_dim || down_out != hidden)
    {
        set_last_error("fused_ffn_act: down weight shape mismatch.");
        return 0;
    }

    const std::size_t ctx_size = 4 * 1024 * 1024;
    PooledContextHandle context;
    if (!context.init(ctx_size))
    {
        set_last_error("fused_ffn_act: failed to create ggml context.");
        return 0;
    }

    std::vector<BufferHandle> host_ptr_buffers;
    bool use_zero_copy = can_map_standard_view(input_desc) && can_map_standard_view(output_desc);

    TensorBinding output_binding;
    TensorBinding input_binding;
    std::vector<float> packed_input;

    if (use_zero_copy)
    {
        ggml_backend_buffer_t out_buf = nullptr;
        ggml_backend_buffer_t inp_buf = nullptr;
        const bool out_ok = create_binding_from_host_ptr_2d(context.value, g_backend, output_desc, output_binding, out_buf);
        const bool inp_ok = out_ok && create_binding_from_host_ptr_2d(context.value, g_backend, input_desc, input_binding, inp_buf);

        if (out_ok && inp_ok)
        {
            host_ptr_buffers.emplace_back(out_buf);
            host_ptr_buffers.emplace_back(inp_buf);
        }
        else
        {
            if (inp_buf != nullptr) ggml_backend_buffer_free(inp_buf);
            if (out_buf != nullptr) ggml_backend_buffer_free(out_buf);
            use_zero_copy = false;
            output_binding = create_standard_binding(context.value, output_desc);
            input_binding = can_map_standard_view(input_desc)
                ? create_standard_binding(context.value, input_desc)
                : create_packed_standard_binding(context.value, input_desc, packed_input);
        }
    }
    else
    {
        output_binding = create_standard_binding(context.value, output_desc);
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

    if (output_binding.storage == nullptr || input_binding.storage == nullptr ||
        norm_w_tensor == nullptr || gate_up_tensor == nullptr || down_tensor == nullptr)
    {
        set_last_error("fused_ffn_act: failed to allocate ggml tensors.");
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

    // Build graph: rms_norm -> mul(gamma) -> gate_up matmul -> act*up -> down matmul -> cpy(output).
    ggml_tensor* contiguous_input = ggml_cont(context.value, input_binding.tensor);
    ggml_tensor* normed = ggml_rms_norm(context.value, contiguous_input, eps);
    ggml_tensor* scaled = ggml_mul(context.value, normed, norm_w_tensor);

    ggml_tensor* scaled_2d = (rows == 1)
        ? ggml_reshape_2d(context.value, scaled, hidden, 1)
        : scaled;

    ggml_tensor* gate_up_mm = ggml_mul_mat(context.value, gate_up_binding_w.tensor, scaled_2d);

    const std::size_t gu_row_bytes = static_cast<std::size_t>(gate_up_out) * sizeof(float);
    const std::size_t half_bytes = static_cast<std::size_t>(half_dim) * sizeof(float);

    ggml_tensor* gate_view = ggml_view_2d(context.value, gate_up_mm, half_dim, rows, gu_row_bytes, 0);
    ggml_tensor* up_view = ggml_view_2d(context.value, gate_up_mm, half_dim, rows, gu_row_bytes, half_bytes);

    // act(gate) * up as one GLU node off the strided halves (see the note in
    // fused_ffn_swiglu_quant_f32_slab): three fewer [rows, half_dim] F32
    // intermediates and three fewer kernel launches.
    ggml_tensor* glu = (act_type == 1)
        ? ggml_geglu_split(context.value, gate_view, up_view)
        : ggml_swiglu_split(context.value, gate_view, up_view);

    ggml_tensor* glu_2d = (rows == 1)
        ? ggml_reshape_2d(context.value, glu, half_dim, 1)
        : glu;

    ggml_tensor* down_mm = ggml_mul_mat(context.value, down_binding_w.tensor, glu_2d);

    ggml_tensor* down_flat = ggml_reshape_1d(context.value, down_mm, static_cast<int64_t>(rows) * hidden);
    ggml_tensor* output_node = ggml_cpy(context.value, down_flat, output_binding.tensor);
    if (output_node == nullptr)
    {
        set_last_error("fused_ffn_act: failed to create output cpy node.");
        return 0;
    }
    ggml_set_output(output_node);
    // The output base is written by the terminal cpy and read back afterwards;
    // keep gallocr from recycling it (see fused_ffn_swiglu_quant_f32_slab).
    ggml_set_output(output_binding.storage);

    ggml_cgraph* graph = ggml_new_graph(context.value);
    if (graph == nullptr)
    {
        set_last_error("fused_ffn_act: failed to create graph.");
        return 0;
    }
    ggml_build_forward_expand(graph, output_node);

    // Persistent, liveness-planning allocator instead of a fresh per-call device
    // buffer (see fused_ffn_swiglu_quant_f32_slab).
    BufferHandle buffer(nullptr);
    if (!alloc_graph_reuse_gallocr(graph))
    {
        buffer.value = ggml_backend_alloc_ctx_tensors(context.value, g_backend);
        if (buffer.value == nullptr)
        {
            set_last_error("fused_ffn_act: failed to allocate backend buffer.");
            return 0;
        }
    }

    host_read_barrier();

    // Output is write-only; never upload it. Upload input + weights as needed.
    if (!use_zero_copy)
    {
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

    ggml_status status = tsg::compute_graph(g_backend, graph);
    if (status != GGML_STATUS_SUCCESS)
    {
        set_last_error("fused_ffn_act: graph execution failed.");
        return 0;
    }
    finalize_compute(use_zero_copy, output_binding.storage, output_desc.data, output_binding.raw_bytes);
    if (buffer.value != nullptr) host_read_barrier();

    clear_last_error();
    return 1;
}

// ============================================================================
// Shared vision-block subgraph builders.
//
// These build the attention / MLP residual subgraph INTO an existing ggml
// context (no compute, no buffer alloc) and return the new residual-stream
// tensor (2D, [hidden, rows] in ggml layout). The single-block fused ops below
// and the whole-encoder fusion (fused_qwen35_vision_encoder_f32_impl) both call
// these, so the per-block correctness test re-validates the exact math used by
// the whole-encoder path.
// ============================================================================

// ----------------------------------------------------------------------------
// Vision-encoder attention: flash on CPU/Metal, explicit F32 SDPA on CUDA and
// Vulkan.
//
// The vision head_dim (72; hidden 1152 / 16 heads for both the Qwen3-VL and
// SigLIP towers) is excluded from every ggml-cuda tensor-core and vec
// flash-attention kernel, so ggml_flash_attn_ext lands on the generic tile
// kernel, which on fast-fp16 NVIDIA GPUs accumulates in F16 and ignores
// GGML_PREC_F32. Over a 4000+ patch bidirectional attention row that loses
// enough precision to visibly corrupt a subset of tokens (measured on the
// Qwen3.5 27-block encoder: per-token cosine vs the CPU backend down to 0.81
// on the worst rows, and the corrupted embeddings derail the language model
// entirely). Padding the KV length to FATTN_KQ_STRIDE and masking, i.e. the
// exact llama.cpp calling convention, reproduces the same bits, so this is
// kernel numerics, not a tail/oob bug.
//
// On CUDA/Vulkan the attention is therefore built as explicit
// mul_mat(F32-prec) + soft_max + mul_mat, with the query axis chunked so the
// [kv, chunk, heads] score tensor stays bounded (~256 MB) instead of the full
// O(patches^2) blow-up that motivated flash here in the first place. The CPU
// and Metal flash kernels compute in F32 and match the reference, so they
// keep the (faster, memory-free) flash path.
//
// Layouts: q_perm/k_perm are [head_dim, rows, heads] permuted views, v_3d is
// [head_dim, heads, rows]. Returns a [head_dim, heads, rows] node, the same
// logical shape ggml_flash_attn_ext produces, so callers cont+reshape
// identically on either path.
// ----------------------------------------------------------------------------
ggml_tensor* build_vision_attention(
    ggml_context* ctx,
    ggml_tensor* q_perm, ggml_tensor* k_perm, ggml_tensor* v_3d,
    int rows, int num_heads, int head_dim, float attn_scale)
{
    if (g_backend_type != BACKEND_TYPE_CUDA && g_backend_type != BACKEND_TYPE_VULKAN)
    {
        ggml_tensor* v_perm = ggml_permute(ctx, v_3d, 0, 2, 1, 3); // [hd, rows, heads]
        return flash_attn_ext_guarded(ctx, "vision attention", q_perm, k_perm, v_perm, nullptr,
            attn_scale, 0.0f, 0.0f, nullptr, GGML_PREC_F32);
    }

    ggml_tensor* q_cont = ggml_cont(ctx, q_perm); // [hd, rows, heads]
    ggml_tensor* k_cont = ggml_cont(ctx, k_perm); // [hd, rows, heads]
    // V transposed for the probs matmul: [rows, hd, heads].
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v_3d, 1, 2, 0, 3));

    // Bound the [rows, chunk, heads] F32 score tensor to ~256 MB per chunk.
    // TS_VISION_ATTN_BUDGET_MB overrides for A/B debugging (e.g. huge = single chunk).
    static const int64_t score_budget_mb = []{
        const char* e = std::getenv("TS_VISION_ATTN_BUDGET_MB");
        return e != nullptr ? std::atoll(e) : 256ll; }();
    const int64_t score_budget_bytes = score_budget_mb * 1024 * 1024;
    int64_t chunk = score_budget_bytes /
        (static_cast<int64_t>(rows) * num_heads * static_cast<int64_t>(sizeof(float)));
    if (chunk < 256) chunk = 256;
    if (chunk > rows) chunk = rows;

    ggml_tensor* out = nullptr; // [hd, rows-so-far, heads]
    for (int64_t start = 0; start < rows; start += chunk)
    {
        const int64_t len = std::min<int64_t>(chunk, rows - start);
        ggml_tensor* q_c = (len == rows)
            ? q_cont
            : ggml_view_3d(ctx, q_cont, head_dim, len, num_heads,
                q_cont->nb[1], q_cont->nb[2], static_cast<std::size_t>(start) * q_cont->nb[1]);
        // Materialize the chunk view before the matmul. ggml-vulkan feeds a
        // non-contiguous F32 src1 through its internal F16 conversion path,
        // which mangles this offset view (ne1 < nb2/nb1) — the encoder output
        // turns into scrambled patches ("glitchy pixels" descriptions) on any
        // image large enough to need more than one chunk. An explicit cont
        // goes through the (correct) CPY op instead, is numerically identical
        // on every backend, and costs a few MB of scratch per chunk.
        if (len != rows)
            q_c = ggml_cont(ctx, q_c);

        ggml_tensor* scores = ggml_mul_mat(ctx, k_cont, q_c); // [rows(kv), len, heads]
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        ggml_tensor* probs = ggml_soft_max_ext(ctx, scores, nullptr, attn_scale, 0.0f);
        ggml_tensor* out_c = ggml_mul_mat(ctx, vt, probs);    // [hd, len, heads]
        out = out == nullptr ? out_c : ggml_concat(ctx, out, out_c, 1);
    }

    return ggml_permute(ctx, out, 0, 2, 1, 3); // [hd, heads, rows]
}

// LN + QKV(+bias) + split + NeoX RoPE(cos/sin tables) + flash-attn + out(+bias) + residual.
static ggml_tensor* build_vision_attn_subgraph(
    ggml_context* ctx, ggml_tensor* cur,
    ggml_tensor* ln_w_t, ggml_tensor* ln_b_t, float eps,
    ggml_tensor* qkv_w_t, ggml_tensor* qkv_b_t,
    ggml_tensor* out_w_t, ggml_tensor* out_b_t,
    ggml_tensor* cos_t, ggml_tensor* sin_t,
    int rows, int hidden, int num_heads, int head_dim, int half_dim,
    float attn_scale)
{
    const int triple_hidden = 3 * hidden;
    ggml_tensor* inp = ggml_cont(ctx, cur);

    // LayerNorm
    ggml_tensor* normed = ggml_norm(ctx, inp, eps);
    ggml_tensor* ln_scaled = ggml_mul(ctx, normed, ln_w_t);
    ggml_tensor* ln_out = ggml_add(ctx, ln_scaled, ln_b_t);

    // QKV projection + bias
    ggml_tensor* ln_2d = (rows == 1) ? ggml_reshape_2d(ctx, ln_out, hidden, 1) : ln_out;
    ggml_tensor* qkv = ggml_mul_mat(ctx, qkv_w_t, ln_2d);
    ggml_tensor* qkv_biased = ggml_add(ctx, qkv, ggml_repeat(ctx, ggml_reshape_2d(ctx, qkv_b_t, triple_hidden, 1), qkv));

    // Split Q, K, V
    std::size_t row_bytes = static_cast<std::size_t>(triple_hidden) * sizeof(float);
    std::size_t d_bytes = static_cast<std::size_t>(hidden) * sizeof(float);
    ggml_tensor* q_raw = ggml_cont(ctx, ggml_view_2d(ctx, qkv_biased, hidden, rows, row_bytes, 0));
    ggml_tensor* k_raw = ggml_cont(ctx, ggml_view_2d(ctx, qkv_biased, hidden, rows, row_bytes, d_bytes));
    ggml_tensor* v_raw = ggml_cont(ctx, ggml_view_2d(ctx, qkv_biased, hidden, rows, row_bytes, 2 * d_bytes));

    ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_raw, head_dim, num_heads, rows);
    ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_raw, head_dim, num_heads, rows);
    ggml_tensor* cos_3d = ggml_reshape_3d(ctx, cos_t, half_dim, 1, rows);
    ggml_tensor* sin_3d = ggml_reshape_3d(ctx, sin_t, half_dim, 1, rows);

    std::size_t head_row_bytes = static_cast<std::size_t>(head_dim) * sizeof(float);
    std::size_t half_bytes_local = static_cast<std::size_t>(half_dim) * sizeof(float);
    auto apply_rope = [&](ggml_tensor* x_3d) -> ggml_tensor* {
        ggml_tensor* x_lo = ggml_view_3d(ctx, x_3d, half_dim, num_heads, rows,
            head_row_bytes, head_row_bytes * num_heads, 0);
        ggml_tensor* x_hi = ggml_view_3d(ctx, x_3d, half_dim, num_heads, rows,
            head_row_bytes, head_row_bytes * num_heads, half_bytes_local);
        ggml_tensor* lo_c = ggml_cont(ctx, x_lo);
        ggml_tensor* hi_c = ggml_cont(ctx, x_hi);
        ggml_tensor* lo_cos = ggml_mul(ctx, lo_c, cos_3d);
        ggml_tensor* hi_sin = ggml_mul(ctx, hi_c, sin_3d);
        ggml_tensor* out_lo = ggml_sub(ctx, lo_cos, hi_sin);
        ggml_tensor* lo_sin = ggml_mul(ctx, lo_c, sin_3d);
        ggml_tensor* hi_cos = ggml_mul(ctx, hi_c, cos_3d);
        ggml_tensor* out_hi = ggml_add(ctx, lo_sin, hi_cos);
        return ggml_concat(ctx, out_lo, out_hi, 0);
    };

    ggml_tensor* q_roped = apply_rope(q_3d);
    ggml_tensor* k_roped = apply_rope(k_3d);
    ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_raw, head_dim, num_heads, rows);

    ggml_tensor* q_perm = ggml_permute(ctx, q_roped, 0, 2, 1, 3);
    ggml_tensor* k_perm = ggml_permute(ctx, k_roped, 0, 2, 1, 3);

    ggml_tensor* attn_out = build_vision_attention(ctx, q_perm, k_perm, v_3d,
        rows, num_heads, head_dim, attn_scale);
    ggml_tensor* attn_flat = ggml_reshape_2d(ctx, ggml_cont(ctx, attn_out), hidden, rows);

    ggml_tensor* out_proj = ggml_mul_mat(ctx, out_w_t, attn_flat);
    ggml_tensor* out_biased = ggml_add(ctx, out_proj, ggml_repeat(ctx, ggml_reshape_2d(ctx, out_b_t, hidden, 1), out_proj));

    // Residual add (2D [hidden, rows] — equivalent to the flattened add).
    return ggml_add(ctx, inp, out_biased);
}

// LN + up(+bias) + GELU + down(+bias) + residual.
static ggml_tensor* build_vision_mlp_subgraph(
    ggml_context* ctx, ggml_tensor* cur,
    ggml_tensor* ln_w_t, ggml_tensor* ln_b_t, float eps,
    ggml_tensor* up_w_t, ggml_tensor* up_b_t,
    ggml_tensor* down_w_t, ggml_tensor* down_b_t,
    int rows, int hidden, int dff)
{
    ggml_tensor* inp = ggml_cont(ctx, cur);

    ggml_tensor* normed = ggml_norm(ctx, inp, eps);
    ggml_tensor* scaled = ggml_mul(ctx, normed, ln_w_t);
    ggml_tensor* ln_out = ggml_add(ctx, scaled, ln_b_t);

    ggml_tensor* ln_2d = (rows == 1) ? ggml_reshape_2d(ctx, ln_out, hidden, 1) : ln_out;
    ggml_tensor* fc1 = ggml_mul_mat(ctx, up_w_t, ln_2d);
    ggml_tensor* fc1_bias = ggml_add(ctx, fc1, ggml_repeat(ctx, ggml_reshape_2d(ctx, up_b_t, dff, 1), fc1));

    ggml_tensor* fc1_gelu = ggml_gelu(ctx, fc1_bias);

    ggml_tensor* fc1_2d = (rows == 1) ? ggml_reshape_2d(ctx, fc1_gelu, dff, 1) : fc1_gelu;
    ggml_tensor* fc2 = ggml_mul_mat(ctx, down_w_t, fc1_2d);
    ggml_tensor* fc2_bias = ggml_add(ctx, fc2, ggml_repeat(ctx, ggml_reshape_2d(ctx, down_b_t, hidden, 1), fc2));

    return ggml_add(ctx, inp, fc2_bias);
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

    // Build the computation graph (shared with the whole-encoder fusion).
    ggml_context* ctx = context.value;
    ggml_tensor* added = build_vision_mlp_subgraph(ctx, hidden_binding.tensor,
        ln_w_t, ln_b_t, eps, up_w_t, up_b_t, down_w_t, down_b_t, rows, hidden, dff);

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

    ggml_status status = tsg::compute_graph(g_backend, graph);
    if (status != GGML_STATUS_SUCCESS)
    {
        set_last_error("fused_vision_mlp: graph compute failed.");
        return 0;
    }
    finalize_compute(use_zero_copy, hidden_binding.storage, hidden_desc.data, hidden_binding.raw_bytes);

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

    // Build graph (shared with the whole-encoder fusion).
    ggml_tensor* added = build_vision_attn_subgraph(ctx, hidden_binding.tensor,
        ln_w_t, ln_b_t, eps, qkv_w_t, qkv_b_t, out_w_t, out_b_t, cos_t, sin_t,
        rows, hidden, num_heads, head_dim, half_dim, attn_scale);

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

    ggml_status status = tsg::compute_graph(g_backend, graph);
    if (status != GGML_STATUS_SUCCESS) { set_last_error("fused_vision_attn: compute failed."); return 0; }
    finalize_compute(use_zero_copy, hidden_binding.storage, hidden_desc.data, hidden_binding.raw_bytes);

    clear_last_error();
    return 1;
}

// ============================================================================
// Fused Gemma-4 SigLIP vision block (attention + gated MLP) in ONE GGML graph.
//
// Mirrors Gemma4VisionEncoder.EncoderBlock (the per-op C# path) exactly:
//   r0 = hidden
//   n1 = rms_norm(r0)*ln1_w
//   q  = clampO(matmul(q_w, clampI(n1)));  k,v likewise   (no bias)
//   q  = rms_norm_perhead(q)*q_norm_w;  k likewise;  v = rms_norm_perhead(v)  (unweighted)
//   q,k = rope2d(q,k)        [NeoX X-rot on first half, Y-rot on second half]
//   a  = flash_attn(q,k,v, scale=1)                       (full bidirectional)
//   o  = clampO(matmul(out_w, clampI(a)))
//   r1 = r0 + rms_norm(o)*attn_post_norm_w                (sandwich norm)
//   n2 = rms_norm(r1)*ln2_w
//   g  = clampO(matmul(gate_w, clampI(n2)));  u = clampO(matmul(up_w, clampI(n2)))
//   h  = gelu_quick(g) * u                                (QuickGELU gated MLP)
//   d  = clampO(matmul(down_w, clampI(h)))
//   out = r1 + rms_norm(d)*ffn_post_norm_w   -> hidden (in place)
//
// clamps[28] = {q,k,v,out,gate,up,down} x {inMin,inMax,outMin,outMax} (QAT
// activation clamps); a |bound| >= 3e38 means "no clamp on that side".
// All weights (F32) are cached on-device by host pointer across encodes; the
// whole block is one upload + one download, replacing ~30 per-op round-trips.
// ============================================================================
int fused_gemma4_vision_block_f32_impl(
    const TensorView2DDesc& hidden_desc,        // [N, D] in/out
    float eps,
    const float* ln1_w,
    const float* q_w, int q_ne0, int q_ne1, std::size_t q_bytes,
    const float* k_w, int k_ne0, int k_ne1, std::size_t k_bytes,
    const float* v_w, int v_ne0, int v_ne1, std::size_t v_bytes,
    const float* q_norm_w, const float* k_norm_w,
    const float* attn_post_norm_w,
    const float* out_w, int out_ne0, int out_ne1, std::size_t out_bytes,
    const float* cosx, const float* sinx, const float* cosy, const float* siny,
    const float* ln2_w,
    const float* gate_w, int gate_ne0, int gate_ne1, std::size_t gate_bytes,
    const float* up_w, int up_ne0, int up_ne1, std::size_t up_bytes,
    const float* down_w, int down_ne0, int down_ne1, std::size_t down_bytes,
    const float* ffn_post_norm_w,
    const float* clamps,
    int num_patches, int num_heads, int head_dim)
{
    if (!ensure_backend())
        return 0;
    if (!validate_desc(hidden_desc, "hidden"))
        return 0;

    const int N  = hidden_desc.dim0;       // num_patches
    const int D  = hidden_desc.dim1;       // hidden_size
    const int hd = head_dim;
    const int nH = num_heads;
    const int quarter = hd / 4;
    const int ff = gate_ne1;               // intermediate_size
    const int cs_elems = num_patches * quarter;

    const std::size_t ctx_size = 16 * 1024 * 1024;
    PooledContextHandle context;
    if (!context.init(ctx_size))
    {
        set_last_error("fused_gemma4_vision_block: context init failed.");
        return 0;
    }
    ggml_context* ctx = context.value;

    std::vector<BufferHandle> host_ptr_buffers;
    bool use_zero_copy = can_map_standard_view(hidden_desc);
    TensorBinding hidden_binding;
    if (use_zero_copy)
    {
        ggml_backend_buffer_t buf = nullptr;
        if (!create_binding_from_host_ptr_2d(ctx, g_backend, hidden_desc, hidden_binding, buf))
        {
            use_zero_copy = false;
            hidden_binding = create_standard_binding(ctx, hidden_desc);
        }
        else
            host_ptr_buffers.emplace_back(buf);
    }
    else
        hidden_binding = create_standard_binding(ctx, hidden_desc);

    // Weights are STREAMED, not cached on-device: the vision tower runs briefly
    // (once per image) and its ~0.5 GB of F32 weights would otherwise pile into
    // VRAM via the persistent device-copy cache, stealing residency from the
    // (much larger) language model and triggering paging that worsens block by
    // block on a VRAM-tight multimodal model (e.g. 26B-A4B). Each weight is a
    // plain graph leaf uploaded fresh into the reused per-graph buffer below.
    struct UP { ggml_tensor* t; const void* data; std::size_t bytes; };
    std::vector<UP> ups;
    auto w1d = [&](const float* d, int n) {
        ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ups.push_back({ t, d, (std::size_t)n * sizeof(float) }); return t; };
    auto w2d = [&](const float* d, int ne0, int ne1, std::size_t bytes) {
        ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
        ups.push_back({ t, d, bytes }); return t; };

    ggml_tensor* ln1_w_t   = w1d(ln1_w, D);
    ggml_tensor* q_w_t     = w2d(q_w, q_ne0, q_ne1, q_bytes);
    ggml_tensor* k_w_t     = w2d(k_w, k_ne0, k_ne1, k_bytes);
    ggml_tensor* v_w_t     = w2d(v_w, v_ne0, v_ne1, v_bytes);
    ggml_tensor* q_norm_t  = w1d(q_norm_w, hd);
    ggml_tensor* k_norm_t  = w1d(k_norm_w, hd);
    ggml_tensor* apn_t     = w1d(attn_post_norm_w, D);
    ggml_tensor* out_w_t   = w2d(out_w, out_ne0, out_ne1, out_bytes);
    ggml_tensor* ln2_w_t   = w1d(ln2_w, D);
    ggml_tensor* gate_w_t  = w2d(gate_w, gate_ne0, gate_ne1, gate_bytes);
    ggml_tensor* up_w_t    = w2d(up_w, up_ne0, up_ne1, up_bytes);
    ggml_tensor* down_w_t  = w2d(down_w, down_ne0, down_ne1, down_bytes);
    ggml_tensor* fpn_t     = w1d(ffn_post_norm_w, D);

    // cos/sin rope tables (small, geometry-dependent): uploaded fresh each call.
    ggml_tensor* cosx_t = w1d(cosx, cs_elems);
    ggml_tensor* sinx_t = w1d(sinx, cs_elems);
    ggml_tensor* cosy_t = w1d(cosy, cs_elems);
    ggml_tensor* siny_t = w1d(siny, cs_elems);

    if (!hidden_binding.storage || !ln1_w_t || !q_w_t || !cosx_t)
    {
        set_last_error("fused_gemma4_vision_block: tensor alloc failed.");
        return 0;
    }

    const float NC = 3.0e38f;
    auto clampt = [&](ggml_tensor* x, float lo, float hi) -> ggml_tensor* {
        if (lo <= -NC && hi >= NC) return x;
        return ggml_clamp(ctx, x, lo, hi);
    };

    ggml_tensor* cosx_3d = ggml_reshape_3d(ctx, cosx_t, quarter, 1, N);
    ggml_tensor* sinx_3d = ggml_reshape_3d(ctx, sinx_t, quarter, 1, N);
    ggml_tensor* cosy_3d = ggml_reshape_3d(ctx, cosy_t, quarter, 1, N);
    ggml_tensor* siny_3d = ggml_reshape_3d(ctx, siny_t, quarter, 1, N);

    // 2D split RoPE: quarters [0:Q]/[Q:2Q] rotated by X angles, [2Q:3Q]/[3Q:4Q] by Y.
    auto rope2d = [&](ggml_tensor* x3 /*[hd, nH, N]*/) -> ggml_tensor* {
        std::size_t nb1 = x3->nb[1], nb2 = x3->nb[2], fb = sizeof(float);
        ggml_tensor* x0 = ggml_cont(ctx, ggml_view_3d(ctx, x3, quarter, nH, N, nb1, nb2, 0));
        ggml_tensor* x1 = ggml_cont(ctx, ggml_view_3d(ctx, x3, quarter, nH, N, nb1, nb2, (std::size_t)quarter * fb));
        ggml_tensor* x2 = ggml_cont(ctx, ggml_view_3d(ctx, x3, quarter, nH, N, nb1, nb2, (std::size_t)2 * quarter * fb));
        ggml_tensor* x3q = ggml_cont(ctx, ggml_view_3d(ctx, x3, quarter, nH, N, nb1, nb2, (std::size_t)3 * quarter * fb));
        ggml_tensor* o0 = ggml_sub(ctx, ggml_mul(ctx, x0, cosx_3d), ggml_mul(ctx, x1, sinx_3d));
        ggml_tensor* o1 = ggml_add(ctx, ggml_mul(ctx, x0, sinx_3d), ggml_mul(ctx, x1, cosx_3d));
        ggml_tensor* o2 = ggml_sub(ctx, ggml_mul(ctx, x2, cosy_3d), ggml_mul(ctx, x3q, siny_3d));
        ggml_tensor* o3 = ggml_add(ctx, ggml_mul(ctx, x2, siny_3d), ggml_mul(ctx, x3q, cosy_3d));
        return ggml_concat(ctx, ggml_concat(ctx, o0, o1, 0), ggml_concat(ctx, o2, o3, 0), 0);
    };

    // ---------------- Build graph ----------------
    ggml_tensor* inp = ggml_cont(ctx, hidden_binding.tensor);   // [D, N]

    // Attention sublayer.
    ggml_tensor* n1 = ggml_mul(ctx, ggml_rms_norm(ctx, inp, eps), ln1_w_t);
    ggml_tensor* q = clampt(ggml_mul_mat(ctx, q_w_t, clampt(n1, clamps[0], clamps[1])), clamps[2], clamps[3]);
    ggml_tensor* k = clampt(ggml_mul_mat(ctx, k_w_t, clampt(n1, clamps[4], clamps[5])), clamps[6], clamps[7]);
    ggml_tensor* v = clampt(ggml_mul_mat(ctx, v_w_t, clampt(n1, clamps[8], clamps[9])), clamps[10], clamps[11]);

    ggml_tensor* qn = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_2d(ctx, q, hd, nH * N), eps), q_norm_t);
    ggml_tensor* kn = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_2d(ctx, k, hd, nH * N), eps), k_norm_t);
    ggml_tensor* vn = ggml_rms_norm(ctx, ggml_reshape_2d(ctx, v, hd, nH * N), eps);  // unweighted

    ggml_tensor* qr = rope2d(ggml_reshape_3d(ctx, qn, hd, nH, N));
    ggml_tensor* kr = rope2d(ggml_reshape_3d(ctx, kn, hd, nH, N));
    ggml_tensor* v3 = ggml_reshape_3d(ctx, vn, hd, nH, N);

    ggml_tensor* qp = ggml_permute(ctx, qr, 0, 2, 1, 3);  // [hd, N, nH]
    ggml_tensor* kp = ggml_permute(ctx, kr, 0, 2, 1, 3);
    ggml_tensor* flash = build_vision_attention(ctx, qp, kp, v3, N, nH, hd, 1.0f);
    if (!backend_supports_op(flash))
    {
        set_last_error("fused_gemma4_vision_block: flash_attn unsupported for this head_dim/backend.");
        return 0;
    }
    ggml_tensor* aflat = ggml_reshape_2d(ctx, ggml_cont(ctx, flash), D, N);
    ggml_tensor* o = clampt(ggml_mul_mat(ctx, out_w_t, clampt(aflat, clamps[12], clamps[13])), clamps[14], clamps[15]);
    ggml_tensor* postA = ggml_mul(ctx, ggml_rms_norm(ctx, o, eps), apn_t);
    ggml_tensor* r1 = ggml_add(ctx, postA, inp);

    // MLP sublayer (QuickGELU gated).
    ggml_tensor* n2 = ggml_mul(ctx, ggml_rms_norm(ctx, r1, eps), ln2_w_t);
    ggml_tensor* g = clampt(ggml_mul_mat(ctx, gate_w_t, clampt(n2, clamps[16], clamps[17])), clamps[18], clamps[19]);
    ggml_tensor* u = clampt(ggml_mul_mat(ctx, up_w_t, clampt(n2, clamps[20], clamps[21])), clamps[22], clamps[23]);
    ggml_tensor* act = ggml_mul(ctx, ggml_gelu_quick(ctx, g), u);
    ggml_tensor* dn = clampt(ggml_mul_mat(ctx, down_w_t, clampt(act, clamps[24], clamps[25])), clamps[26], clamps[27]);
    ggml_tensor* postF = ggml_mul(ctx, ggml_rms_norm(ctx, dn, eps), fpn_t);
    ggml_tensor* outv = ggml_add(ctx, postF, r1);   // normed first: lets ggml-metal fuse rms_norm+mul+add into one kernel

    ggml_tensor* output = ggml_cpy(ctx, outv, hidden_binding.tensor);
    if (!output)
    {
        set_last_error("fused_gemma4_vision_block: output cpy failed.");
        return 0;
    }
    ggml_set_output(output);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 8192, false);
    if (!graph) { set_last_error("fused_gemma4_vision_block: graph failed."); return 0; }
    ggml_build_forward_expand(graph, output);

    // Allocate the per-graph intermediates (and the streamed weight leaves) into
    // the persistent, reused gallocr buffer rather than a fresh device buffer per
    // block. A per-call alloc+free of this ~40 MB scratch fragments device VRAM
    // across the tower's blocks and gets progressively slower on a near-full GPU;
    // the reused buffer is grown once and kept. Falls back to per-call allocation
    // when the reuse path is unavailable (disabled / non-Metal-or-CUDA backend).
    BufferHandle buffer(nullptr);
    if (!alloc_graph_reuse_gallocr(graph))
    {
        buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
        if (!buffer.value) { set_last_error("fused_gemma4_vision_block: buffer alloc failed."); return 0; }
    }

    // Drain the previous block's GPU work before overwriting the shared reused
    // buffer (no-op on the eager-sync path).
    host_read_barrier();

    if (!use_zero_copy)
        upload_binding(hidden_binding, hidden_desc.data, hidden_binding.raw_bytes);
    for (auto& u : ups)
        ggml_backend_tensor_set(u.t, u.data, 0, u.bytes);

    ggml_status status = tsg::compute_graph(g_backend, graph);
    if (status != GGML_STATUS_SUCCESS) { set_last_error("fused_gemma4_vision_block: compute failed."); return 0; }
    finalize_compute(use_zero_copy, hidden_binding.storage, hidden_desc.data, hidden_binding.raw_bytes);
    // Drain the queued async download before the per-call fallback buffer frees
    // (no-op on the reuse-gallocr path, where buffer.value == nullptr).
    if (buffer.value != nullptr) host_read_barrier();

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
    // One GLU node straight off the strided halves (see the note in
    // fused_ffn_swiglu_quant_f32_slab): saves three [rows, half_dim] F32
    // intermediates and three kernel launches per layer.
    ggml_tensor* gate_v = ggml_view_2d(ctx, gu_mm, half_dim, rows, gu_row_bytes, 0);
    ggml_tensor* up_v   = ggml_view_2d(ctx, gu_mm, half_dim, rows, gu_row_bytes, half_bytes);
    ggml_tensor* swiglu = ggml_swiglu_split(ctx, gate_v, up_v);
    ggml_tensor* swiglu_2d = (rows == 1) ? ggml_reshape_2d(ctx, swiglu, half_dim, 1) : swiglu;
    ggml_tensor* dn_mm = ggml_mul_mat(ctx, dn_w, swiglu_2d);

    ggml_tensor* dn_flat = ggml_reshape_1d(ctx, dn_mm, static_cast<int64_t>(rows) * hidden);
    ggml_tensor* res_flat2 = ggml_reshape_1d(ctx, res_2d, static_cast<int64_t>(rows) * hidden);
    ggml_tensor* final_res = ggml_add(ctx, res_flat2, dn_flat);

    ggml_tensor* output = ggml_cpy(ctx, final_res, residual_binding.tensor);
    if (!output) { set_last_error("fused_outproj_ffn: output cpy failed."); return 0; }
    ggml_set_output(output);
    // Keep gallocr from recycling the residual base: it is read at the start and
    // written by the terminal cpy (see fused_ffn_swiglu_quant_f32_slab).
    ggml_set_output(residual_binding.storage);

    ggml_cgraph* graph = ggml_new_graph(ctx);
    if (!graph) { set_last_error("fused_outproj_ffn: graph failed."); return 0; }
    ggml_build_forward_expand(graph, output);

    // Persistent, liveness-planning allocator instead of a fresh per-call device
    // buffer (see fused_ffn_swiglu_quant_f32_slab). This graph is the larger of
    // the two: it also carries the GDN value-dim input, which is what asked for
    // 645 MiB in a single cudaMalloc on the 1073-token multimodal prefill.
    BufferHandle buffer(nullptr);
    if (!alloc_graph_reuse_gallocr(graph))
    {
        buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
        if (!buffer.value) { set_last_error("fused_outproj_ffn: buffer alloc failed."); return 0; }
    }

    host_read_barrier();

    if (!use_zero_copy) {
        upload_binding(residual_binding, residual_desc.data, residual_binding.raw_bytes);
        if (packed_input.empty()) upload_binding(input_binding, input_desc.data, input_binding.raw_bytes);
        else upload_binding(input_binding, packed_input.data(), input_binding.raw_bytes);
    }

    if (!out_bound || out_upl) upload_binding(out_w_bind, out_proj_quant.data, out_w_bind.raw_bytes);
    if (!gu_bound || gu_upl) upload_binding(gu_w_bind, gate_up_quant.data, gu_w_bind.raw_bytes);
    if (!dn_bound || dn_upl) upload_binding(dn_w_bind, down_quant.data, dn_w_bind.raw_bytes);
    if (!norm_bound || norm_upl) { TensorBinding tmp = { norm_w, norm_w, norm_bytes }; upload_binding(tmp, ffn_norm_weight_data, norm_bytes); }

    ggml_status status = tsg::compute_graph(g_backend, graph);
    if (status != GGML_STATUS_SUCCESS) { set_last_error("fused_outproj_ffn: compute failed."); return 0; }
    finalize_compute(use_zero_copy, residual_binding.storage, residual_desc.data, residual_binding.raw_bytes);
    if (buffer.value != nullptr) host_read_barrier();

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

    ggml_status status = tsg::compute_graph(g_backend, graph);
    if (status != GGML_STATUS_SUCCESS) { set_last_error("fused_outproj_norm_router: compute failed."); return 0; }
    // Multi-tensor download path. Drain any prior async work, then sync+download.
    host_read_barrier();
    tsg::sync_backend(g_backend);

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
    std::int64_t m2_raw_bytes,
    // Tensor parallelism: with tp_degree > 1 `m2` is this rank's row-parallel
    // shard and the call returns a plan cut at the matmul instead of running it.
    int tp_degree, void** tp_plan_out)
{
    try
    {
        QuantizedWeightDesc m2_quant;
        m2_quant.data = m2_data;
        m2_quant.ggml_type = m2_ggml_type;
        m2_quant.ne0 = m2_ne0;
        m2_quant.ne1 = m2_ne1;
        m2_quant.raw_bytes = m2_raw_bytes;
        return fused_matmul_quant_add_f32_impl(residual, input, m2_quant, tp_degree, tp_plan_out);
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
    int half_dim,
    // Tensor parallelism: with tp_degree > 1 the weights are this rank's shards
    // (gate_up column-parallel, down row-parallel) and the call returns a plan
    // cut at the down projection instead of running the graph.
    int tp_degree, void** tp_plan_out)
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
            gate_up_quant, down_quant, half_dim, tp_degree, tp_plan_out);
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

// Fused dense FFN projection (norm + gate_up + act*up + down) -> output, no
// residual fold. act_type: 0 = SiLU (SwiGLU), 1 = GELU tanh (GeGLU, Gemma 4).
TSG_EXPORT int TSGgml_FusedFFNActProjectQuantF32(
    TensorView2DDesc output,
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
    int half_dim,
    int act_type)
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

        return fused_ffn_act_project_quant_f32_impl(
            output, input, norm_weight_data, norm_weight_count, eps,
            gate_up_quant, down_quant, half_dim, act_type);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown error in fused FFN act-project.");
        return 0;
    }
}

// ============================================================================
// Fused RMSNorm + residual add in one GGML graph:
//   residual = residual + rms_norm(input, normW, eps)
//
// Gemma applies this "post-norm then residual add" pattern 3x per layer
// (post-attention, post-FFN, post-PLE). On the GGML backend each was previously
// two dispatches (Ops.RMSNorm + Ops.Add); this collapses them to one graph,
// halving the dispatch + host round-trip count for those ops. residual and input
// are distinct [rows, hidden] F32 tensors. Mirrors the MLX
// (MlxFusedOps.TryRmsNormAddInPlace) and pure-CUDA (CudaFusedOps.TryRmsNormResidualAdd)
// fused paths that the GGML backend lacked.
// ============================================================================
int fused_rms_norm_residual_add_f32_impl(
    const TensorView2DDesc& residual_desc,
    const TensorView2DDesc& input_desc,
    void* norm_weight_data,
    int norm_weight_count,
    float eps)
{
    if (!ensure_backend())
        return 0;
    if (!validate_desc(residual_desc, "residual") || !validate_desc(input_desc, "input"))
        return 0;
    if (norm_weight_data == nullptr || norm_weight_count <= 0)
    {
        set_last_error("fused_rms_norm_add: invalid norm weight.");
        return 0;
    }

    const int rows = input_desc.dim0;
    const int hidden = input_desc.dim1;
    if (residual_desc.dim0 != rows || residual_desc.dim1 != hidden)
    {
        set_last_error("fused_rms_norm_add: residual/input shape mismatch.");
        return 0;
    }
    if (norm_weight_count != hidden)
    {
        set_last_error("fused_rms_norm_add: norm_weight_count != hidden.");
        return 0;
    }

    const std::size_t ctx_size = 1024 * 1024;
    PooledContextHandle context;
    if (!context.init(ctx_size))
    {
        set_last_error("fused_rms_norm_add: failed to create ggml context.");
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
    if (residual_binding.storage == nullptr || input_binding.storage == nullptr || norm_w_tensor == nullptr)
    {
        set_last_error("fused_rms_norm_add: failed to allocate ggml tensors.");
        return 0;
    }

    // Norm weight host-ptr cache (stable pointer across calls).
    bool norm_bound = false, norm_needs_upload = false;
    std::size_t norm_bytes = static_cast<std::size_t>(norm_weight_count) * sizeof(float);
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        if (dev != nullptr && norm_bytes >= 4096)
        {
            ggml_backend_buffer_t buf = nullptr;
            void* addr = nullptr;
            if (try_get_cacheable_tensor_buffer(g_backend, dev, norm_w_tensor, norm_weight_data, norm_bytes, buf, addr, norm_needs_upload))
            {
                ggml_status st = ggml_backend_tensor_alloc(buf, norm_w_tensor, addr);
                norm_bound = (st == GGML_STATUS_SUCCESS);
                if (!norm_bound) invalidate_cached_buffer(norm_weight_data);
            }
        }
    }

    ggml_tensor* contiguous_input = ggml_cont(context.value, input_binding.tensor);
    ggml_tensor* contiguous_residual = ggml_cont(context.value, residual_binding.tensor);
    ggml_tensor* normed = ggml_rms_norm(context.value, contiguous_input, eps);
    ggml_tensor* scaled = ggml_mul(context.value, normed, norm_w_tensor);
    ggml_tensor* scaled_flat = ggml_reshape_1d(context.value, scaled, static_cast<int64_t>(rows) * hidden);
    ggml_tensor* res_flat = ggml_reshape_1d(context.value, contiguous_residual, static_cast<int64_t>(rows) * hidden);
    ggml_tensor* added = ggml_add(context.value, res_flat, scaled_flat);
    ggml_tensor* output = ggml_cpy(context.value, added, residual_binding.tensor);
    if (output == nullptr)
    {
        set_last_error("fused_rms_norm_add: failed to create output cpy node.");
        return 0;
    }
    ggml_set_output(output);

    ggml_cgraph* graph = ggml_new_graph(context.value);
    ggml_build_forward_expand(graph, output);

    BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
    if (buffer.value == nullptr)
    {
        set_last_error("fused_rms_norm_add: failed to allocate backend buffer.");
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
    if (!norm_bound || norm_needs_upload)
    {
        TensorBinding tmp = { norm_w_tensor, norm_w_tensor, norm_bytes };
        upload_binding(tmp, norm_weight_data, norm_bytes);
    }

    ggml_status status = tsg::compute_graph(g_backend, graph);
    if (status != GGML_STATUS_SUCCESS)
    {
        set_last_error("fused_rms_norm_add: graph execution failed.");
        return 0;
    }
    finalize_compute(use_zero_copy, residual_binding.storage, residual_desc.data, residual_binding.raw_bytes);

    clear_last_error();
    return 1;
}

TSG_EXPORT int TSGgml_FusedRmsNormResidualAddF32(
    TensorView2DDesc residual,
    TensorView2DDesc input,
    void* norm_weight_data,
    int norm_weight_count,
    float eps)
{
    try
    {
        return fused_rms_norm_residual_add_f32_impl(residual, input, norm_weight_data, norm_weight_count, eps);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown error in fused rms_norm residual add.");
        return 0;
    }
}

// ============================================================================
// Fused Gemma Per-Layer-Embeddings (PLE) block in one GGML graph:
//   residual += rms_norm( proj_W^T @ ( gelu(inp_gate_W^T @ residual) * perLayerInput ),
//                         post_norm_W, eps )
//
// Collapses the PLE chain (inp_gate matmul → GELU·mul(perLayerInput) → proj matmul
// → post_norm → residual add) — previously 4 GGML dispatches (the norm+add was
// already fused) — into ONE graph, keeping the small [ple_dim, rows] intermediate
// on-device. inp_gate_W is [hidden, ple_dim], proj_W is [ple_dim, hidden];
// perLayerInput is [rows, ple_dim] F32. `residual` is both the inp_gate matmul
// input and the accumulator — read via a contiguous copy made at graph head, so
// the final cpy-back can't race the matmul read. gelu is the tanh approximation
// (Gemma's gelu_pytorch_tanh), matching Ops.GELUMul.
// ============================================================================
int fused_ple_block_quant_f32_impl(
    const TensorView2DDesc& residual_desc,
    const TensorView2DDesc& per_layer_input_desc,
    const QuantizedWeightDesc& inp_gate_quant,
    const QuantizedWeightDesc& proj_quant,
    void* post_norm_data,
    int post_norm_count,
    float eps)
{
    if (!ensure_backend())
        return 0;
    if (!validate_desc(residual_desc, "residual") || !validate_desc(per_layer_input_desc, "per_layer_input"))
        return 0;
    if (post_norm_data == nullptr || post_norm_count <= 0)
    {
        set_last_error("fused_ple: invalid post-norm weight.");
        return 0;
    }
    if (inp_gate_quant.data == nullptr || inp_gate_quant.ne0 <= 0 || inp_gate_quant.ne1 <= 0 || inp_gate_quant.raw_bytes <= 0 ||
        proj_quant.data == nullptr || proj_quant.ne0 <= 0 || proj_quant.ne1 <= 0 || proj_quant.raw_bytes <= 0)
    {
        set_last_error("fused_ple: invalid weight descriptor.");
        return 0;
    }

    const int rows = residual_desc.dim0;
    const int hidden = residual_desc.dim1;
    const int ple_dim = per_layer_input_desc.dim1;
    if (per_layer_input_desc.dim0 != rows)
    {
        set_last_error("fused_ple: per_layer_input row count mismatch.");
        return 0;
    }
    if (inp_gate_quant.ne0 != hidden || inp_gate_quant.ne1 != ple_dim ||
        proj_quant.ne0 != ple_dim || proj_quant.ne1 != hidden || post_norm_count != hidden)
    {
        set_last_error("fused_ple: weight shape mismatch.");
        return 0;
    }

    const std::size_t ctx_size = 2 * 1024 * 1024;
    PooledContextHandle context;
    if (!context.init(ctx_size))
    {
        set_last_error("fused_ple: failed to create ggml context.");
        return 0;
    }

    std::vector<BufferHandle> host_ptr_buffers;
    bool use_zero_copy = can_map_standard_view(residual_desc) && can_map_standard_view(per_layer_input_desc);

    TensorBinding residual_binding;
    TensorBinding pli_binding;
    std::vector<float> packed_pli;

    if (use_zero_copy)
    {
        ggml_backend_buffer_t res_buf = nullptr;
        ggml_backend_buffer_t pli_buf = nullptr;
        const bool res_ok = create_binding_from_host_ptr_2d(context.value, g_backend, residual_desc, residual_binding, res_buf);
        const bool pli_ok = res_ok && create_binding_from_host_ptr_2d(context.value, g_backend, per_layer_input_desc, pli_binding, pli_buf);
        if (res_ok && pli_ok)
        {
            host_ptr_buffers.emplace_back(res_buf);
            host_ptr_buffers.emplace_back(pli_buf);
        }
        else
        {
            if (pli_buf != nullptr) ggml_backend_buffer_free(pli_buf);
            if (res_buf != nullptr) ggml_backend_buffer_free(res_buf);
            use_zero_copy = false;
            residual_binding = create_standard_binding(context.value, residual_desc);
            pli_binding = can_map_standard_view(per_layer_input_desc)
                ? create_standard_binding(context.value, per_layer_input_desc)
                : create_packed_standard_binding(context.value, per_layer_input_desc, packed_pli);
        }
    }
    else
    {
        residual_binding = create_standard_binding(context.value, residual_desc);
        pli_binding = can_map_standard_view(per_layer_input_desc)
            ? create_standard_binding(context.value, per_layer_input_desc)
            : create_packed_standard_binding(context.value, per_layer_input_desc, packed_pli);
    }

    ggml_tensor* post_norm_tensor = ggml_new_tensor_1d(context.value, GGML_TYPE_F32, post_norm_count);
    ggml_type inp_gate_type = static_cast<ggml_type>(inp_gate_quant.ggml_type);
    ggml_tensor* inp_gate_tensor = ggml_new_tensor_2d(context.value, inp_gate_type, inp_gate_quant.ne0, inp_gate_quant.ne1);
    TensorBinding inp_gate_binding = { inp_gate_tensor, inp_gate_tensor, static_cast<std::size_t>(inp_gate_quant.raw_bytes) };
    ggml_type proj_type = static_cast<ggml_type>(proj_quant.ggml_type);
    ggml_tensor* proj_tensor = ggml_new_tensor_2d(context.value, proj_type, proj_quant.ne0, proj_quant.ne1);
    TensorBinding proj_binding = { proj_tensor, proj_tensor, static_cast<std::size_t>(proj_quant.raw_bytes) };

    if (residual_binding.storage == nullptr || pli_binding.storage == nullptr ||
        post_norm_tensor == nullptr || inp_gate_tensor == nullptr || proj_tensor == nullptr)
    {
        set_last_error("fused_ple: failed to allocate ggml tensors.");
        return 0;
    }

    auto try_cache_quant = [](ggml_tensor* t, const QuantizedWeightDesc& q, bool& bound, bool& needs_upload) {
        bound = false; needs_upload = false;
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        if (dev == nullptr || q.raw_bytes < 4096) return;
        ggml_backend_buffer_t buf = nullptr; void* addr = nullptr;
        if (try_get_cacheable_tensor_buffer(g_backend, dev, t, q.data, static_cast<std::size_t>(q.raw_bytes), buf, addr, needs_upload))
        {
            ggml_status st = ggml_backend_tensor_alloc(buf, t, addr);
            bound = (st == GGML_STATUS_SUCCESS);
            if (!bound) invalidate_cached_buffer(q.data);
        }
    };
    bool inp_gate_bound = false, inp_gate_needs_upload = false;
    try_cache_quant(inp_gate_tensor, inp_gate_quant, inp_gate_bound, inp_gate_needs_upload);
    bool proj_bound = false, proj_needs_upload = false;
    try_cache_quant(proj_tensor, proj_quant, proj_bound, proj_needs_upload);

    bool norm_bound = false, norm_needs_upload = false;
    std::size_t norm_bytes = static_cast<std::size_t>(post_norm_count) * sizeof(float);
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        if (dev != nullptr && norm_bytes >= 4096)
        {
            ggml_backend_buffer_t buf = nullptr; void* addr = nullptr;
            if (try_get_cacheable_tensor_buffer(g_backend, dev, post_norm_tensor, post_norm_data, norm_bytes, buf, addr, norm_needs_upload))
            {
                ggml_status st = ggml_backend_tensor_alloc(buf, post_norm_tensor, addr);
                norm_bound = (st == GGML_STATUS_SUCCESS);
                if (!norm_bound) invalidate_cached_buffer(post_norm_data);
            }
        }
    }

    // Build graph.
    ggml_tensor* contiguous_residual = ggml_cont(context.value, residual_binding.tensor);
    ggml_tensor* res_2d = (rows == 1)
        ? ggml_reshape_2d(context.value, contiguous_residual, hidden, 1)
        : contiguous_residual;
    ggml_tensor* gate = ggml_mul_mat(context.value, inp_gate_binding.tensor, res_2d); // [ple_dim, rows]
    ggml_mul_mat_set_prec(gate, GGML_PREC_F32);
    ggml_tensor* gate_gelu = ggml_gelu(context.value, gate);
    ggml_tensor* glu = ggml_mul(context.value, gate_gelu, pli_binding.tensor);        // [ple_dim, rows]
    ggml_tensor* glu_2d = (rows == 1)
        ? ggml_reshape_2d(context.value, glu, ple_dim, 1)
        : glu;
    ggml_tensor* ple_proj = ggml_mul_mat(context.value, proj_binding.tensor, glu_2d); // [hidden, rows]
    ggml_mul_mat_set_prec(ple_proj, GGML_PREC_F32);
    ggml_tensor* normed = ggml_rms_norm(context.value, ple_proj, eps);
    ggml_tensor* scaled = ggml_mul(context.value, normed, post_norm_tensor);
    ggml_tensor* scaled_flat = ggml_reshape_1d(context.value, scaled, static_cast<int64_t>(rows) * hidden);
    ggml_tensor* res_flat = ggml_reshape_1d(context.value, contiguous_residual, static_cast<int64_t>(rows) * hidden);
    ggml_tensor* added = ggml_add(context.value, res_flat, scaled_flat);
    ggml_tensor* output = ggml_cpy(context.value, added, residual_binding.tensor);
    if (output == nullptr)
    {
        set_last_error("fused_ple: failed to create output cpy node.");
        return 0;
    }
    ggml_set_output(output);

    ggml_cgraph* graph = ggml_new_graph(context.value);
    ggml_build_forward_expand(graph, output);

    BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
    if (buffer.value == nullptr)
    {
        set_last_error("fused_ple: failed to allocate backend buffer.");
        return 0;
    }

    if (!use_zero_copy)
    {
        upload_binding(residual_binding, residual_desc.data, residual_binding.raw_bytes);
        if (packed_pli.empty())
            upload_binding(pli_binding, per_layer_input_desc.data, pli_binding.raw_bytes);
        else
            upload_binding(pli_binding, packed_pli.data(), pli_binding.raw_bytes);
    }
    if (!inp_gate_bound || inp_gate_needs_upload)
        upload_binding(inp_gate_binding, inp_gate_quant.data, inp_gate_binding.raw_bytes);
    if (!proj_bound || proj_needs_upload)
        upload_binding(proj_binding, proj_quant.data, proj_binding.raw_bytes);
    if (!norm_bound || norm_needs_upload)
    {
        TensorBinding tmp = { post_norm_tensor, post_norm_tensor, norm_bytes };
        upload_binding(tmp, post_norm_data, norm_bytes);
    }

    ggml_status status = tsg::compute_graph(g_backend, graph);
    if (status != GGML_STATUS_SUCCESS)
    {
        set_last_error("fused_ple: graph execution failed.");
        return 0;
    }
    finalize_compute(use_zero_copy, residual_binding.storage, residual_desc.data, residual_binding.raw_bytes);

    clear_last_error();
    return 1;
}

TSG_EXPORT int TSGgml_FusedPleBlockQuantF32(
    TensorView2DDesc residual,
    TensorView2DDesc per_layer_input,
    void* inp_gate_data, int inp_gate_ggml_type, std::int64_t inp_gate_ne0, std::int64_t inp_gate_ne1, std::int64_t inp_gate_raw_bytes,
    void* proj_data, int proj_ggml_type, std::int64_t proj_ne0, std::int64_t proj_ne1, std::int64_t proj_raw_bytes,
    void* post_norm_data, int post_norm_count, float eps)
{
    try
    {
        QuantizedWeightDesc inp_gate_quant;
        inp_gate_quant.data = inp_gate_data;
        inp_gate_quant.ggml_type = inp_gate_ggml_type;
        inp_gate_quant.ne0 = inp_gate_ne0;
        inp_gate_quant.ne1 = inp_gate_ne1;
        inp_gate_quant.raw_bytes = inp_gate_raw_bytes;

        QuantizedWeightDesc proj_quant;
        proj_quant.data = proj_data;
        proj_quant.ggml_type = proj_ggml_type;
        proj_quant.ne0 = proj_ne0;
        proj_quant.ne1 = proj_ne1;
        proj_quant.raw_bytes = proj_raw_bytes;

        return fused_ple_block_quant_f32_impl(
            residual, per_layer_input, inp_gate_quant, proj_quant, post_norm_data, post_norm_count, eps);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown error in fused PLE block.");
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

// ============================================================================
// Whole vision encoder (Qwen3.5/3.6-VL): ALL transformer blocks as ONE device
// graph. The per-block fused ops (above) each issue a separate synchronous
// graph_compute and round-trip the [N, hidden] residual over PCIe; for 27
// blocks that is 54 dispatches at ~10% GPU utilisation. This runs every block
// back-to-back in a single graph with the residual kept device-resident
// (gallocr liveness-reuse bounds the activation peak) and ONE sync at the end.
//
// Weight pointer arrays are block-major (index [b]); all blocks share identical
// shapes. Math is byte-identical to the per-block path (shared subgraph
// builders), so the per-block correctness checksum re-validates this path.
// ============================================================================
static int fused_qwen35_vision_encoder_f32_impl(
    const TensorView2DDesc& hidden_desc,
    int block_count, float eps, float attn_scale,
    int num_patches, int num_heads, int head_dim, int half_dim,
    const float* cos_table, const float* sin_table,
    const float* const* ln1_w, const float* const* ln1_b,
    const float* const* qkv_w, const float* const* qkv_b,
    const float* const* out_w, const float* const* out_b,
    const float* const* ln2_w, const float* const* ln2_b,
    const float* const* up_w,  const float* const* up_b,
    const float* const* down_w, const float* const* down_b,
    int ln_dim,
    int qkv_ne0, int qkv_ne1, std::size_t qkv_bytes, int qkv_b_dim,
    int out_ne0, int out_ne1, std::size_t out_bytes, int out_b_dim,
    int up_ne0, int up_ne1, std::size_t up_bytes, int up_b_dim,
    int down_ne0, int down_ne1, std::size_t down_bytes, int down_b_dim)
{
    if (!ensure_backend()) return 0;
    if (!validate_desc(hidden_desc, "hidden")) return 0;
    if (block_count <= 0 || block_count > 128) { set_last_error("vision_encoder: bad block_count"); return 0; }

    const int rows = hidden_desc.dim0;     // numPatches
    const int hidden = hidden_desc.dim1;   // hiddenSize
    const int dff = up_ne1;                // intermediate_size
    const std::size_t cos_sin_elems = static_cast<std::size_t>(num_patches) * half_dim;

    // ctx holds only tensor metadata (no_alloc) + the graph node array. ~block_count
    // blocks x ~42 nodes + ~12 weight leaves ≈ <1 MB even at 128 blocks; 32 MB is the
    // pool's max slot size.
    const std::size_t ctx_size = static_cast<std::size_t>(32) * 1024 * 1024;
    PooledContextHandle context;
    if (!context.init(ctx_size)) { set_last_error("vision_encoder: ctx init failed."); return 0; }
    ggml_context* ctx = context.value;

    // Residual-stream binding (zero-copy host-mapped when possible).
    std::vector<BufferHandle> host_ptr_buffers;
    bool use_zero_copy = can_map_standard_view(hidden_desc);
    TensorBinding hidden_binding;
    if (use_zero_copy)
    {
        ggml_backend_buffer_t buf = nullptr;
        if (!create_binding_from_host_ptr_2d(ctx, g_backend, hidden_desc, hidden_binding, buf))
        {
            use_zero_copy = false;
            hidden_binding = create_standard_binding(ctx, hidden_desc);
        }
        else
            host_ptr_buffers.emplace_back(buf);
    }
    else
        hidden_binding = create_standard_binding(ctx, hidden_desc);
    if (!hidden_binding.storage) { set_last_error("vision_encoder: hidden bind failed."); return 0; }

    // cos/sin tables: shared across all blocks. Uploaded after gallocr alloc, so
    // mark as inputs (gallocr never reuses input memory → upload stays valid).
    ggml_tensor* cos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cos_sin_elems);
    ggml_tensor* sin_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cos_sin_elems);
    if (!cos_t || !sin_t) { set_last_error("vision_encoder: cos/sin alloc failed."); return 0; }
    ggml_set_input(cos_t);
    ggml_set_input(sin_t);

    ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
    struct HostBinding { ggml_tensor* tensor; const void* data; std::size_t bytes; };
    std::vector<HostBinding> upload_list;
    // Cache a weight device-resident (own buffer, persists across encodes); on a
    // cache miss bind it as a gallocr input leaf and upload after alloc.
    auto bind_w = [&](ggml_tensor* t, const void* data, std::size_t bytes) {
        if (t == nullptr || data == nullptr) return;
        if (bytes >= 4096 && dev != nullptr)
        {
            ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool need = false;
            if (try_get_cacheable_tensor_buffer(g_backend, dev, t, const_cast<void*>(data), bytes, buf, addr, need))
            {
                if (ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS)
                {
                    if (need) upload_list.push_back({ t, data, bytes });
                    return;
                }
                invalidate_cached_buffer(const_cast<void*>(data));
            }
        }
        ggml_set_input(t);
        upload_list.push_back({ t, data, bytes });
    };

    ggml_tensor* cur = hidden_binding.tensor;
    for (int b = 0; b < block_count; b++)
    {
        ggml_tensor* ln1w_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ln_dim);
        ggml_tensor* ln1b_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ln_dim);
        ggml_tensor* qkvw_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, qkv_ne0, qkv_ne1);
        ggml_tensor* qkvb_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, qkv_b_dim);
        ggml_tensor* outw_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, out_ne0, out_ne1);
        ggml_tensor* outb_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_b_dim);
        ggml_tensor* ln2w_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ln_dim);
        ggml_tensor* ln2b_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ln_dim);
        ggml_tensor* upw_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, up_ne0, up_ne1);
        ggml_tensor* upb_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, up_b_dim);
        ggml_tensor* downw_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, down_ne0, down_ne1);
        ggml_tensor* downb_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, down_b_dim);
        if (!ln1w_t || !qkvw_t || !outw_t || !ln2w_t || !upw_t || !downw_t)
        { set_last_error("vision_encoder: block tensor alloc failed."); return 0; }

        bind_w(ln1w_t, ln1_w[b], static_cast<std::size_t>(ln_dim) * sizeof(float));
        bind_w(ln1b_t, ln1_b[b], static_cast<std::size_t>(ln_dim) * sizeof(float));
        bind_w(qkvw_t, qkv_w[b], qkv_bytes);
        bind_w(qkvb_t, qkv_b[b], static_cast<std::size_t>(qkv_b_dim) * sizeof(float));
        bind_w(outw_t, out_w[b], out_bytes);
        bind_w(outb_t, out_b[b], static_cast<std::size_t>(out_b_dim) * sizeof(float));
        bind_w(ln2w_t, ln2_w[b], static_cast<std::size_t>(ln_dim) * sizeof(float));
        bind_w(ln2b_t, ln2_b[b], static_cast<std::size_t>(ln_dim) * sizeof(float));
        bind_w(upw_t, up_w[b], up_bytes);
        bind_w(upb_t, up_b[b], static_cast<std::size_t>(up_b_dim) * sizeof(float));
        bind_w(downw_t, down_w[b], down_bytes);
        bind_w(downb_t, down_b[b], static_cast<std::size_t>(down_b_dim) * sizeof(float));

        cur = build_vision_attn_subgraph(ctx, cur, ln1w_t, ln1b_t, eps,
            qkvw_t, qkvb_t, outw_t, outb_t, cos_t, sin_t,
            rows, hidden, num_heads, head_dim, half_dim, attn_scale);
        cur = build_vision_mlp_subgraph(ctx, cur, ln2w_t, ln2b_t, eps,
            upw_t, upb_t, downw_t, downb_t, rows, hidden, dff);
    }

    ggml_tensor* output = ggml_cpy(ctx, cur, hidden_binding.tensor);
    if (!output) { set_last_error("vision_encoder: output cpy failed."); return 0; }
    ggml_set_output(output);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, static_cast<std::size_t>(block_count) * 200 + 512, false);
    if (!graph) { set_last_error("vision_encoder: graph creation failed."); return 0; }
    ggml_build_forward_expand(graph, output);

    // Dedicated gallocr (freed below): liveness-reuse bounds the activation peak
    // to ~max-concurrent (not the 27-block sum). Freed each encode so the encoder
    // buffer is not held resident while the language model decodes.
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
    if (galloc == nullptr || !ggml_gallocr_alloc_graph(galloc, graph))
    {
        if (galloc != nullptr) ggml_gallocr_free(galloc);
        set_last_error("vision_encoder: gallocr allocation failed.");
        return 0;
    }

    if (!use_zero_copy)
        upload_binding(hidden_binding, hidden_desc.data, hidden_binding.raw_bytes);
    for (auto& u : upload_list)
        ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);
    ggml_backend_tensor_set(cos_t, cos_table, 0, cos_sin_elems * sizeof(float));
    ggml_backend_tensor_set(sin_t, sin_table, 0, cos_sin_elems * sizeof(float));

    ggml_status status = tsg::compute_graph(g_backend, graph);
    if (status != GGML_STATUS_SUCCESS) { ggml_gallocr_free(galloc); set_last_error("vision_encoder: graph compute failed."); return 0; }
    finalize_compute(use_zero_copy, hidden_binding.storage, hidden_desc.data, hidden_binding.raw_bytes);

    ggml_gallocr_free(galloc);
    clear_last_error();
    return 1;
}


// ---------------------------------------------------------------------------
// GLM-5.3-Flash (glm5next) vision encoder: the GLM-OCR ViT as ONE graph.
// Differences from the Qwen3.5 tower above: RMS norms without biases, per-head
// RMS q/k norms (one [head_dim] weight shared by every head), and a SwiGLU-clamp
// MLP (gate <= L, up in [-L, L]) instead of the GELU 2-layer MLP. The rope, the
// attention core and the whole-graph mechanics are shared.
// ---------------------------------------------------------------------------

static ggml_tensor* build_glm_vision_attn_subgraph(
    ggml_context* ctx, ggml_tensor* cur,
    ggml_tensor* ln_w_t, float eps,
    ggml_tensor* qkv_w_t, ggml_tensor* qkv_b_t,
    ggml_tensor* qn_w_t, ggml_tensor* kn_w_t,
    ggml_tensor* out_w_t, ggml_tensor* out_b_t,
    ggml_tensor* cos_t, ggml_tensor* sin_t,
    int rows, int hidden, int num_heads, int head_dim, int half_dim,
    float attn_scale)
{
    const int triple_hidden = 3 * hidden;
    ggml_tensor* inp = ggml_cont(ctx, cur);

    // RMS norm (no bias)
    ggml_tensor* ln_out = ggml_mul(ctx, ggml_rms_norm(ctx, inp, eps), ln_w_t);

    ggml_tensor* qkv = ggml_mul_mat(ctx, qkv_w_t, ln_out);
    ggml_tensor* qkv_biased = ggml_add(ctx, qkv, ggml_repeat(ctx, ggml_reshape_2d(ctx, qkv_b_t, triple_hidden, 1), qkv));

    std::size_t row_bytes = static_cast<std::size_t>(triple_hidden) * sizeof(float);
    std::size_t d_bytes = static_cast<std::size_t>(hidden) * sizeof(float);
    ggml_tensor* q_raw = ggml_cont(ctx, ggml_view_2d(ctx, qkv_biased, hidden, rows, row_bytes, 0));
    ggml_tensor* k_raw = ggml_cont(ctx, ggml_view_2d(ctx, qkv_biased, hidden, rows, row_bytes, d_bytes));
    ggml_tensor* v_raw = ggml_cont(ctx, ggml_view_2d(ctx, qkv_biased, hidden, rows, row_bytes, 2 * d_bytes));

    ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_raw, head_dim, num_heads, rows);
    ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_raw, head_dim, num_heads, rows);

    // per-head RMS q/k norms: rms_norm normalizes dim 0 (= head_dim) and the
    // [head_dim] weight broadcasts across heads and rows.
    q_3d = ggml_mul(ctx, ggml_rms_norm(ctx, q_3d, eps), qn_w_t);
    k_3d = ggml_mul(ctx, ggml_rms_norm(ctx, k_3d, eps), kn_w_t);

    ggml_tensor* cos_3d = ggml_reshape_3d(ctx, cos_t, half_dim, 1, rows);
    ggml_tensor* sin_3d = ggml_reshape_3d(ctx, sin_t, half_dim, 1, rows);

    std::size_t head_row_bytes = static_cast<std::size_t>(head_dim) * sizeof(float);
    std::size_t half_bytes_local = static_cast<std::size_t>(half_dim) * sizeof(float);
    auto apply_rope = [&](ggml_tensor* x_3d) -> ggml_tensor* {
        ggml_tensor* x_lo = ggml_view_3d(ctx, x_3d, half_dim, num_heads, rows,
            head_row_bytes, head_row_bytes * num_heads, 0);
        ggml_tensor* x_hi = ggml_view_3d(ctx, x_3d, half_dim, num_heads, rows,
            head_row_bytes, head_row_bytes * num_heads, half_bytes_local);
        ggml_tensor* lo_c = ggml_cont(ctx, x_lo);
        ggml_tensor* hi_c = ggml_cont(ctx, x_hi);
        ggml_tensor* out_lo = ggml_sub(ctx, ggml_mul(ctx, lo_c, cos_3d), ggml_mul(ctx, hi_c, sin_3d));
        ggml_tensor* out_hi = ggml_add(ctx, ggml_mul(ctx, lo_c, sin_3d), ggml_mul(ctx, hi_c, cos_3d));
        return ggml_concat(ctx, out_lo, out_hi, 0);
    };

    // The q/k-norm outputs are non-contiguous mul results over reshaped views;
    // cont them so the rope's views land on plain layouts.
    ggml_tensor* q_roped = apply_rope(ggml_cont(ctx, q_3d));
    ggml_tensor* k_roped = apply_rope(ggml_cont(ctx, k_3d));
    ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_raw, head_dim, num_heads, rows);

    ggml_tensor* q_perm = ggml_permute(ctx, q_roped, 0, 2, 1, 3);
    ggml_tensor* k_perm = ggml_permute(ctx, k_roped, 0, 2, 1, 3);

    ggml_tensor* attn_out = build_vision_attention(ctx, q_perm, k_perm, v_3d,
        rows, num_heads, head_dim, attn_scale);
    ggml_tensor* attn_flat = ggml_reshape_2d(ctx, ggml_cont(ctx, attn_out), hidden, rows);

    ggml_tensor* out_proj = ggml_mul_mat(ctx, out_w_t, attn_flat);
    ggml_tensor* out_biased = ggml_add(ctx, out_proj, ggml_repeat(ctx, ggml_reshape_2d(ctx, out_b_t, hidden, 1), out_proj));

    return ggml_add(ctx, inp, out_biased);
}

static ggml_tensor* build_glm_vision_mlp_subgraph(
    ggml_context* ctx, ggml_tensor* cur,
    ggml_tensor* ln_w_t, float eps,
    ggml_tensor* gate_w_t, ggml_tensor* gate_b_t,
    ggml_tensor* up_w_t, ggml_tensor* up_b_t,
    ggml_tensor* down_w_t, ggml_tensor* down_b_t,
    int rows, int hidden, int dff, float swiglu_limit)
{
    ggml_tensor* inp = ggml_cont(ctx, cur);
    ggml_tensor* ln_out = ggml_mul(ctx, ggml_rms_norm(ctx, inp, eps), ln_w_t);

    ggml_tensor* gate = ggml_mul_mat(ctx, gate_w_t, ln_out);
    gate = ggml_add(ctx, gate, ggml_repeat(ctx, ggml_reshape_2d(ctx, gate_b_t, dff, 1), gate));
    ggml_tensor* up = ggml_mul_mat(ctx, up_w_t, ln_out);
    up = ggml_add(ctx, up, ggml_repeat(ctx, ggml_reshape_2d(ctx, up_b_t, dff, 1), up));

    if (swiglu_limit > 0.0f)
    {
        gate = ggml_clamp(ctx, gate, -INFINITY, swiglu_limit);
        up = ggml_clamp(ctx, up, -swiglu_limit, swiglu_limit);
    }
    ggml_tensor* h = ggml_mul(ctx, ggml_silu(ctx, gate), up);

    ggml_tensor* down = ggml_mul_mat(ctx, down_w_t, h);
    down = ggml_add(ctx, down, ggml_repeat(ctx, ggml_reshape_2d(ctx, down_b_t, hidden, 1), down));

    return ggml_add(ctx, inp, down);
}

static int fused_glm_vision_encoder_f32_impl(
    const TensorView2DDesc& hidden_desc,
    int block_count, float eps, float attn_scale, float swiglu_limit,
    int num_patches, int num_heads, int head_dim, int half_dim,
    const float* cos_table, const float* sin_table,
    const float* const* ln1_w,
    const float* const* qkv_w, const float* const* qkv_b,
    const float* const* qn_w, const float* const* kn_w,
    const float* const* out_w, const float* const* out_b,
    const float* const* ln2_w,
    const float* const* gate_w, const float* const* gate_b,
    const float* const* up_w,  const float* const* up_b,
    const float* const* down_w, const float* const* down_b,
    int ln_dim,
    int qkv_ne0, int qkv_ne1, std::size_t qkv_bytes,
    int out_ne0, int out_ne1, std::size_t out_bytes,
    int ffn_ne0, int ffn_ne1, std::size_t ffn_up_bytes, std::size_t ffn_down_bytes)
{
    if (!ensure_backend()) return 0;
    if (!validate_desc(hidden_desc, "hidden")) return 0;
    if (block_count <= 0 || block_count > 128) { set_last_error("glm_vision_encoder: bad block_count"); return 0; }

    const int rows = hidden_desc.dim0;
    const int hidden = hidden_desc.dim1;
    const int dff = ffn_ne1;
    const std::size_t cos_sin_elems = static_cast<std::size_t>(num_patches) * half_dim;

    const std::size_t ctx_size = static_cast<std::size_t>(32) * 1024 * 1024;
    PooledContextHandle context;
    if (!context.init(ctx_size)) { set_last_error("glm_vision_encoder: ctx init failed."); return 0; }
    ggml_context* ctx = context.value;

    std::vector<BufferHandle> host_ptr_buffers;
    bool use_zero_copy = can_map_standard_view(hidden_desc);
    TensorBinding hidden_binding;
    if (use_zero_copy)
    {
        ggml_backend_buffer_t buf = nullptr;
        if (!create_binding_from_host_ptr_2d(ctx, g_backend, hidden_desc, hidden_binding, buf))
        {
            use_zero_copy = false;
            hidden_binding = create_standard_binding(ctx, hidden_desc);
        }
        else
            host_ptr_buffers.emplace_back(buf);
    }
    else
        hidden_binding = create_standard_binding(ctx, hidden_desc);
    if (!hidden_binding.storage) { set_last_error("glm_vision_encoder: hidden bind failed."); return 0; }

    ggml_tensor* cos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cos_sin_elems);
    ggml_tensor* sin_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cos_sin_elems);
    if (!cos_t || !sin_t) { set_last_error("glm_vision_encoder: cos/sin alloc failed."); return 0; }
    ggml_set_input(cos_t);
    ggml_set_input(sin_t);

    ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
    struct HostBinding { ggml_tensor* tensor; const void* data; std::size_t bytes; };
    std::vector<HostBinding> upload_list;
    auto bind_w = [&](ggml_tensor* t, const void* data, std::size_t bytes) {
        if (t == nullptr || data == nullptr) return;
        if (bytes >= 4096 && dev != nullptr)
        {
            ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool need = false;
            if (try_get_cacheable_tensor_buffer(g_backend, dev, t, const_cast<void*>(data), bytes, buf, addr, need))
            {
                if (ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS)
                {
                    if (need) upload_list.push_back({ t, data, bytes });
                    return;
                }
                invalidate_cached_buffer(const_cast<void*>(data));
            }
        }
        ggml_set_input(t);
        upload_list.push_back({ t, data, bytes });
    };

    ggml_tensor* cur = hidden_binding.tensor;
    for (int b = 0; b < block_count; b++)
    {
        ggml_tensor* ln1w_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ln_dim);
        ggml_tensor* qkvw_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, qkv_ne0, qkv_ne1);
        ggml_tensor* qkvb_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, qkv_ne1);
        ggml_tensor* qnw_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
        ggml_tensor* knw_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
        ggml_tensor* outw_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, out_ne0, out_ne1);
        ggml_tensor* outb_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_ne1);
        ggml_tensor* ln2w_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ln_dim);
        ggml_tensor* gatew_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ffn_ne0, ffn_ne1);
        ggml_tensor* gateb_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ffn_ne1);
        ggml_tensor* upw_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ffn_ne0, ffn_ne1);
        ggml_tensor* upb_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ffn_ne1);
        ggml_tensor* downw_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ffn_ne1, ffn_ne0);
        ggml_tensor* downb_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ffn_ne0);
        if (!ln1w_t || !qkvw_t || !outw_t || !ln2w_t || !gatew_t || !upw_t || !downw_t)
        { set_last_error("glm_vision_encoder: block tensor alloc failed."); return 0; }

        bind_w(ln1w_t, ln1_w[b], static_cast<std::size_t>(ln_dim) * sizeof(float));
        bind_w(qkvw_t, qkv_w[b], qkv_bytes);
        bind_w(qkvb_t, qkv_b[b], static_cast<std::size_t>(qkv_ne1) * sizeof(float));
        bind_w(qnw_t, qn_w[b], static_cast<std::size_t>(head_dim) * sizeof(float));
        bind_w(knw_t, kn_w[b], static_cast<std::size_t>(head_dim) * sizeof(float));
        bind_w(outw_t, out_w[b], out_bytes);
        bind_w(outb_t, out_b[b], static_cast<std::size_t>(out_ne1) * sizeof(float));
        bind_w(ln2w_t, ln2_w[b], static_cast<std::size_t>(ln_dim) * sizeof(float));
        bind_w(gatew_t, gate_w[b], ffn_up_bytes);
        bind_w(gateb_t, gate_b[b], static_cast<std::size_t>(ffn_ne1) * sizeof(float));
        bind_w(upw_t, up_w[b], ffn_up_bytes);
        bind_w(upb_t, up_b[b], static_cast<std::size_t>(ffn_ne1) * sizeof(float));
        bind_w(downw_t, down_w[b], ffn_down_bytes);
        bind_w(downb_t, down_b[b], static_cast<std::size_t>(ffn_ne0) * sizeof(float));

        cur = build_glm_vision_attn_subgraph(ctx, cur, ln1w_t, eps,
            qkvw_t, qkvb_t, qnw_t, knw_t, outw_t, outb_t, cos_t, sin_t,
            rows, hidden, num_heads, head_dim, half_dim, attn_scale);
        cur = build_glm_vision_mlp_subgraph(ctx, cur, ln2w_t, eps,
            gatew_t, gateb_t, upw_t, upb_t, downw_t, downb_t, rows, hidden, dff, swiglu_limit);
    }

    ggml_tensor* output = ggml_cpy(ctx, cur, hidden_binding.tensor);
    if (!output) { set_last_error("glm_vision_encoder: output cpy failed."); return 0; }
    ggml_set_output(output);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, static_cast<std::size_t>(block_count) * 220 + 512, false);
    if (!graph) { set_last_error("glm_vision_encoder: graph creation failed."); return 0; }
    ggml_build_forward_expand(graph, output);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
    if (galloc == nullptr || !ggml_gallocr_alloc_graph(galloc, graph))
    {
        if (galloc != nullptr) ggml_gallocr_free(galloc);
        set_last_error("glm_vision_encoder: gallocr allocation failed.");
        return 0;
    }

    if (!use_zero_copy)
        upload_binding(hidden_binding, hidden_desc.data, hidden_binding.raw_bytes);
    for (auto& u : upload_list)
        ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);
    ggml_backend_tensor_set(cos_t, cos_table, 0, cos_sin_elems * sizeof(float));
    ggml_backend_tensor_set(sin_t, sin_table, 0, cos_sin_elems * sizeof(float));

    ggml_status status = tsg::compute_graph(g_backend, graph);
    if (status != GGML_STATUS_SUCCESS) { ggml_gallocr_free(galloc); set_last_error("glm_vision_encoder: graph compute failed."); return 0; }
    finalize_compute(use_zero_copy, hidden_binding.storage, hidden_desc.data, hidden_binding.raw_bytes);

    ggml_gallocr_free(galloc);
    clear_last_error();
    return 1;
}

TSG_EXPORT int TSGgml_GlmVisionEncoderF32(
    TensorView2DDesc hidden,
    int block_count, float eps, float attn_scale, float swiglu_limit,
    int num_patches, int num_heads, int head_dim, int half_dim,
    const float* cos_table, const float* sin_table,
    const float* const* ln1_w,
    const float* const* qkv_w, const float* const* qkv_b,
    const float* const* qn_w, const float* const* kn_w,
    const float* const* out_w, const float* const* out_b,
    const float* const* ln2_w,
    const float* const* gate_w, const float* const* gate_b,
    const float* const* up_w, const float* const* up_b,
    const float* const* down_w, const float* const* down_b,
    int ln_dim,
    int qkv_ne0, int qkv_ne1, int64_t qkv_bytes,
    int out_ne0, int out_ne1, int64_t out_bytes,
    int ffn_ne0, int ffn_ne1, int64_t ffn_up_bytes, int64_t ffn_down_bytes)
{
    try
    {
        return fused_glm_vision_encoder_f32_impl(hidden, block_count, eps, attn_scale, swiglu_limit,
            num_patches, num_heads, head_dim, half_dim, cos_table, sin_table,
            ln1_w, qkv_w, qkv_b, qn_w, kn_w, out_w, out_b, ln2_w,
            gate_w, gate_b, up_w, up_b, down_w, down_b,
            ln_dim,
            qkv_ne0, qkv_ne1, static_cast<std::size_t>(qkv_bytes),
            out_ne0, out_ne1, static_cast<std::size_t>(out_bytes),
            ffn_ne0, ffn_ne1, static_cast<std::size_t>(ffn_up_bytes), static_cast<std::size_t>(ffn_down_bytes));
    }
    catch (const std::exception& e) { set_last_error(e.what()); return 0; }
    catch (...) { set_last_error("glm_vision_encoder: unknown error"); return 0; }
}

TSG_EXPORT int TSGgml_Qwen35VisionEncoderF32(
    TensorView2DDesc hidden,
    int block_count, float eps, float attn_scale,
    int num_patches, int num_heads, int head_dim, int half_dim,
    const float* cos_table, const float* sin_table,
    const float* const* ln1_w, const float* const* ln1_b,
    const float* const* qkv_w, const float* const* qkv_b,
    const float* const* out_w, const float* const* out_b,
    const float* const* ln2_w, const float* const* ln2_b,
    const float* const* up_w,  const float* const* up_b,
    const float* const* down_w, const float* const* down_b,
    int ln_dim,
    int qkv_ne0, int qkv_ne1, std::int64_t qkv_bytes, int qkv_b_dim,
    int out_ne0, int out_ne1, std::int64_t out_bytes, int out_b_dim,
    int up_ne0, int up_ne1, std::int64_t up_bytes, int up_b_dim,
    int down_ne0, int down_ne1, std::int64_t down_bytes, int down_b_dim)
{
    try {
        return fused_qwen35_vision_encoder_f32_impl(hidden, block_count, eps, attn_scale,
            num_patches, num_heads, head_dim, half_dim, cos_table, sin_table,
            ln1_w, ln1_b, qkv_w, qkv_b, out_w, out_b, ln2_w, ln2_b, up_w, up_b, down_w, down_b,
            ln_dim,
            qkv_ne0, qkv_ne1, static_cast<std::size_t>(qkv_bytes), qkv_b_dim,
            out_ne0, out_ne1, static_cast<std::size_t>(out_bytes), out_b_dim,
            up_ne0, up_ne1, static_cast<std::size_t>(up_bytes), up_b_dim,
            down_ne0, down_ne1, static_cast<std::size_t>(down_bytes), down_b_dim);
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in fused vision encoder."); return 0; }
}

TSG_EXPORT int TSGgml_FusedGemma4VisionBlockF32(
    TensorView2DDesc hidden, float eps,
    const float* ln1_w,
    const float* q_w, int q_ne0, int q_ne1, std::int64_t q_bytes,
    const float* k_w, int k_ne0, int k_ne1, std::int64_t k_bytes,
    const float* v_w, int v_ne0, int v_ne1, std::int64_t v_bytes,
    const float* q_norm_w, const float* k_norm_w,
    const float* attn_post_norm_w,
    const float* out_w, int out_ne0, int out_ne1, std::int64_t out_bytes,
    const float* cosx, const float* sinx, const float* cosy, const float* siny,
    const float* ln2_w,
    const float* gate_w, int gate_ne0, int gate_ne1, std::int64_t gate_bytes,
    const float* up_w, int up_ne0, int up_ne1, std::int64_t up_bytes,
    const float* down_w, int down_ne0, int down_ne1, std::int64_t down_bytes,
    const float* ffn_post_norm_w,
    const float* clamps,
    int num_patches, int num_heads, int head_dim)
{
    try {
        return fused_gemma4_vision_block_f32_impl(hidden, eps, ln1_w,
            q_w, q_ne0, q_ne1, static_cast<std::size_t>(q_bytes),
            k_w, k_ne0, k_ne1, static_cast<std::size_t>(k_bytes),
            v_w, v_ne0, v_ne1, static_cast<std::size_t>(v_bytes),
            q_norm_w, k_norm_w, attn_post_norm_w,
            out_w, out_ne0, out_ne1, static_cast<std::size_t>(out_bytes),
            cosx, sinx, cosy, siny, ln2_w,
            gate_w, gate_ne0, gate_ne1, static_cast<std::size_t>(gate_bytes),
            up_w, up_ne0, up_ne1, static_cast<std::size_t>(up_bytes),
            down_w, down_ne0, down_ne1, static_cast<std::size_t>(down_bytes),
            ffn_post_norm_w, clamps, num_patches, num_heads, head_dim);
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in fused gemma4 vision block."); return 0; }
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

        ggml_status status = tsg::compute_graph(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("MoE: graph compute failed");
            return 0;
        }

        finalize_compute(zc, res_bind.storage, result.data, res_bind.raw_bytes);

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
// Batched MoE SwiGLU expert forward: standard SwiGLU activation pattern.
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

        ggml_status status = tsg::compute_graph(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("MoE SwiGLU: graph compute failed");
            return 0;
        }

        finalize_compute(zc, res_bind.storage, result.data, res_bind.raw_bytes);

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

        ggml_status status = tsg::compute_graph(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("MoE SwiGLU residual: graph compute failed");
            return 0;
        }

        finalize_compute(zc, res_bind.storage, residual.data, res_bind.raw_bytes);

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
