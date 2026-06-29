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

        finalize_compute(use_zero_copy, result_binding.storage, result_desc.data, result_binding.raw_bytes);

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

            // Multi-tensor download path (gradients). We drain any pending async work
            // so the explicit downloads below run on a quiesced backend, and clear
            // the pending flag because we're about to ggml_backend_tensor_get
            // (which would force a sync anyway).
            host_read_barrier();
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

        // Multi-tensor download path (gradients). Drain pending async work first
        // so the explicit downloads below run on a quiesced backend.
        host_read_barrier();
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

        finalize_compute_with_download(result_binding.storage, result_desc.data, result_binding.raw_bytes);

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
        bool invert_positions,
        const float* freq_factors,
        int freq_factors_len)
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

        const bool has_freq_factors = freq_factors != nullptr && freq_factors_len > 0;
        if (has_freq_factors && freq_factors_len != rope_dim / 2)
        {
            set_last_error("rope_ex freq_factors length must equal rope_dim/2.");
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
        ggml_tensor* freq_factors_tensor = has_freq_factors
            ? ggml_new_tensor_1d(context.value, GGML_TYPE_F32, freq_factors_len)
            : nullptr;
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr ||
            position_tensor == nullptr ||
            (has_freq_factors && freq_factors_tensor == nullptr))
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
            freq_factors_tensor,
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
        if (has_freq_factors)
        {
            ggml_backend_tensor_set(freq_factors_tensor, freq_factors, 0,
                static_cast<std::size_t>(freq_factors_len) * sizeof(float));
        }

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

        finalize_compute_with_download(result_binding.storage, result_desc.data, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    // Native MRoPE: wraps ggml_rope_multi (kernel_rope_multi.metal). Unlike
    // rope_ex which flattens [seqLen, numHeads] into ne[2], MRoPE keeps them
    // separate so the per-axis position table is indexed by token (not by
    // token*head row). Input shape is [headDim, numHeads, seqLen, 1] and the
    // positions tensor is length 4*seqLen with per-axis-concatenated layout
    //   pos[ 0 ..  n-1 ] = T axis positions (one per token)
    //   pos[ n .. 2n-1 ] = H axis positions
    //   pos[2n .. 3n-1 ] = W axis positions
    //   pos[3n .. 4n-1 ] = 4th-axis positions (zeros for static images)
    // matching ggml.c:4081's `a->ne[2] * 4 == b->ne[0]` and Metal
    // kernel_rope_multi's `pos[i2 + ne02 * axis]` indexing.
    int rope_mrope_f32_impl(
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& src_desc,
        const ContiguousTensorDesc& positions_desc,
        int rope_dim,
        int mode,
        const int sections[4],
        int original_context_length,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow)
    {
        if (!ensure_backend())
            return 0;

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src") || !validate_desc(positions_desc, "positions"))
            return 0;

        if (!same_shape(result_desc, src_desc))
        {
            set_last_error("Source tensor shape does not match result shape for ggml rope_mrope.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml rope_mrope Metal path.");
            return 0;
        }

        if (rope_dim <= 0 || rope_dim > src_desc.ne0 || (rope_dim % 2) != 0)
        {
            set_last_error("rope_dim must be positive, even, and within the source embedding dimension.");
            return 0;
        }

        // Expect 4-D input shape [headDim, numHeads, seqLen, 1].
        const std::int64_t headDim  = src_desc.ne0;
        const std::int64_t numHeads = src_desc.ne1;
        const std::int64_t seqLen   = src_desc.ne2;
        const std::int64_t batch    = src_desc.ne3;
        if (batch != 1)
        {
            set_last_error("rope_mrope expects batch=1 input.");
            return 0;
        }
        if (positions_desc.element_count != 4 * seqLen)
        {
            set_last_error("rope_mrope positions length must be 4 * seqLen.");
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
        TensorBinding src_binding    = create_standard_binding(context.value, src_desc);
        ggml_tensor* position_tensor = ggml_new_tensor_1d(context.value, GGML_TYPE_I32, 4 * seqLen);
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr ||
            position_tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors.");
            return 0;
        }

        // Use the existing 4D shape (headDim, numHeads, seqLen, 1). The
        // ggml_rope_multi assert is a->ne[2] * 4 == b->ne[0], i.e.
        // seqLen * 4 == positions length, which we already enforce above.
        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        if (contiguous_src == nullptr)
        {
            set_last_error("Failed to make ggml rope_mrope source contiguous.");
            return 0;
        }

        std::vector<std::int32_t> positions;
        if (!read_i32_values(positions, positions_desc, "positions"))
            return 0;

        int sections_local[4] = { sections[0], sections[1], sections[2], sections[3] };

        ggml_tensor* rope_tensor = ggml_rope_multi(
            context.value,
            contiguous_src,
            position_tensor,
            /*freq_factors=*/nullptr,
            rope_dim,
            sections_local,
            mode,
            original_context_length,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow);

        if (rope_tensor == nullptr)
        {
            set_last_error("Failed to create ggml rope_multi node.");
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, rope_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml rope_mrope output copy node.");
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
        ggml_backend_tensor_set(position_tensor, positions.data(), 0,
            positions.size() * sizeof(std::int32_t));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        finalize_compute_with_download(result_binding.storage, result_desc.data, result_binding.raw_bytes);
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

        finalize_compute(use_zero_copy, result_binding.storage, result_desc.data, result_binding.raw_bytes);

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

        finalize_compute(use_zero_copy, result_binding.storage, result_desc.data, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    // ----------------------------------------------------------------------
    // attention_softmax_with_sinks_f32_impl
    //
    // Fused causal + sliding-window mask + softmax + attention sinks for
    // GPT-OSS-style attention. Replaces three separate C# ops that used to
    // shuttle the scores tensor between GPU and CPU:
    //   - Ops.AddCausalMask(scores, ...)     (GPU)
    //   - ApplySWAMask(scores, ...)          (CPU walk on device memory)
    //   - ApplySoftmaxWithSinks(scores, ...) (CPU walk on device memory,
    //     ~6 billion MathF.Exp calls per pp2048 forward pass)
    //
    // The CPU softmax-with-sinks is the dominant cost in GPT-OSS prefill
    // (~76% of total time on pp2048), and forces a sync mid-attention because
    // it reads device-resident scores via GetFloatPtr. Doing it on GPU lets the
    // entire attention block stay async.
    //
    // Layout: scores is C# [numHeads, seqLen, kvLen] which maps to GGML
    // [kvLen, seqLen, numHeads] (head_dim-fastest convention).
    // sinks (optional) is a [numHeads] F32 array; null disables sinks.
    // The causal mask blocks any kv index k > maskStartPos + q (autoregressive).
    // The SWA mask additionally blocks any k < maskStartPos + q - slidingWindow + 1
    // when slidingWindow > 0; pass slidingWindow <= 0 to disable.
    int attention_softmax_with_sinks_f32_impl(
        const TensorView3DDesc& scores_desc,   // [numHeads, seqLen, kvLen] - in-place
        const float* sinks_data,                // [numHeads] or nullptr
        int num_heads,
        int seq_len,
        int kv_len,
        int mask_start_pos,
        int sliding_window,
        float scale)
    {
        if (!ensure_backend()) return 0;
        if (!validate_desc(scores_desc, "scores")) return 0;

        if (scores_desc.dim0 != num_heads ||
            scores_desc.dim1 != seq_len ||
            scores_desc.dim2 != kv_len)
        {
            set_last_error("scores tensor shape doesn't match (num_heads, seq_len, kv_len) in SoftmaxWithSinks.");
            return 0;
        }

        if (num_heads <= 0 || seq_len <= 0 || kv_len <= 0)
        {
            set_last_error("Invalid (num_heads, seq_len, kv_len) for SoftmaxWithSinks.");
            return 0;
        }

        if (!can_map_standard_view(scores_desc))
        {
            set_last_error("scores layout is not supported by the SoftmaxWithSinks Metal path.");
            return 0;
        }

        const std::size_t ctx_size = 4 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context for SoftmaxWithSinks.");
            return 0;
        }
        auto* ctx = context.value;

        // Bind scores zero-copy (in-place).
        bool use_zero_copy = true;
        std::vector<BufferHandle> host_ptr_buffers;
        TensorBinding scores_binding;
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_3d(ctx, g_backend, scores_desc, scores_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
            scores_binding = create_standard_binding(ctx, scores_desc);

        if (scores_binding.tensor == nullptr || scores_binding.storage == nullptr)
        {
            set_last_error("Failed to allocate scores binding in SoftmaxWithSinks.");
            return 0;
        }

        // Build the causal+SWA mask on host as F16 [kv_len, seq_len, 1, 1].
        // ggml_soft_max_ext broadcasts the mask across the head dimension.
        ggml_tensor* mask_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv_len, seq_len, 1, 1);
        if (mask_tensor == nullptr)
        {
            set_last_error("Failed to allocate mask tensor in SoftmaxWithSinks.");
            return 0;
        }

        std::vector<ggml_fp16_t> mask_data(static_cast<std::size_t>(kv_len) * seq_len);
        {
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
            for (int q_idx = 0; q_idx < seq_len; q_idx++)
            {
                int q_pos = mask_start_pos + q_idx;
                int win_start = (sliding_window > 0) ? std::max(0, q_pos - sliding_window + 1) : 0;
                ggml_fp16_t* row = &mask_data[static_cast<std::size_t>(q_idx) * kv_len];
                for (int kv_idx = 0; kv_idx < kv_len; kv_idx++)
                    row[kv_idx] = (kv_idx > q_pos || kv_idx < win_start) ? neg_inf : zero_val;
            }
        }

        // Optional sinks tensor [num_heads]. ggml_soft_max_add_sinks treats it as
        // an extra column in the per-row softmax denominator.
        ggml_tensor* sinks_tensor = nullptr;
        if (sinks_data != nullptr)
        {
            sinks_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_heads);
            if (sinks_tensor == nullptr)
            {
                set_last_error("Failed to allocate sinks tensor in SoftmaxWithSinks.");
                return 0;
            }
        }

        // Build the graph:
        //   scores' = soft_max_ext(scores, mask, scale, max_bias=0)
        //   if sinks: soft_max_add_sinks(scores', sinks)
        //   cpy scores' back to caller's scores buffer.
        //
        // soft_max_ext requires a contiguous source tensor; we copy first because
        // the host_ptr binding may be a view with non-trivial strides.
        ggml_tensor* scores_input = ggml_cont(ctx, scores_binding.tensor);
        ggml_tensor* sm = ggml_soft_max_ext(ctx, scores_input, mask_tensor, scale, 0.0f);
        if (sm == nullptr)
        {
            set_last_error("Failed to create soft_max_ext node in SoftmaxWithSinks.");
            return 0;
        }
        if (sinks_tensor != nullptr)
            ggml_soft_max_add_sinks(sm, sinks_tensor);

        ggml_tensor* output = ggml_cpy(ctx, sm, scores_binding.tensor);
        if (output == nullptr)
        {
            set_last_error("Failed to create output cpy node in SoftmaxWithSinks.");
            return 0;
        }
        ggml_set_output(output);

        ggml_cgraph* graph = ggml_new_graph(ctx);
        if (graph == nullptr)
        {
            set_last_error("Failed to create graph in SoftmaxWithSinks.");
            return 0;
        }
        ggml_build_forward_expand(graph, output);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate backend buffer in SoftmaxWithSinks.");
            return 0;
        }

        if (!use_zero_copy)
            upload_binding(scores_binding, scores_desc.data, scores_binding.raw_bytes);

        // Drain pending async work so the upcoming CPU memcpys are safe (mask is
        // built fresh on the C++ stack so it doesn't race; sinks_data is a host
        // array passed by the caller and might point to a tensor-allocated buffer
        // that the previous async op wrote to).
        host_read_barrier();

        ggml_backend_tensor_set(mask_tensor, mask_data.data(), 0,
            mask_data.size() * sizeof(ggml_fp16_t));
        if (sinks_tensor != nullptr)
            ggml_backend_tensor_set(sinks_tensor, sinks_data, 0,
                static_cast<std::size_t>(num_heads) * sizeof(float));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("graph compute failed in SoftmaxWithSinks.");
            return 0;
        }

        finalize_compute(use_zero_copy, scores_binding.storage, scores_desc.data,
            scores_binding.raw_bytes);

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

        finalize_compute(use_zero_copy, result_binding.storage, result_desc.data, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

} // anonymous namespace

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
            invert_positions != 0,
            nullptr,
            0);
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

TSG_EXPORT int TSGgml_RoPEExFreqFactorsF32(
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
    int invert_positions,
    const float* freq_factors,
    int freq_factors_len)
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
            invert_positions != 0,
            freq_factors,
            freq_factors_len);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml rope_ex_ff failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_RoPEMRoPEF32(
    TensorView4DDesc result,
    TensorView4DDesc src,
    ContiguousTensorDesc positions,
    int rope_dim,
    int mode,
    int sect0, int sect1, int sect2, int sect3,
    int original_context_length,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow)
{
    try
    {
        int sections[4] = { sect0, sect1, sect2, sect3 };
        return rope_mrope_f32_impl(
            result,
            src,
            positions,
            rope_dim,
            mode,
            sections,
            original_context_length,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml rope_mrope failure.");
        return 0;
    }
}

// ============================================================================
// Session cache for the head-first fused prefill attention graph.
//
// Chunked prefill of a 42-layer Gemma model issues hundreds of fused-prefill
// attention calls per forward (≈ numLayers × numChunks). Each call otherwise
// builds a fresh ggml graph AND a fresh backend buffer
// (ggml_backend_alloc_ctx_tensors) — pure per-call overhead that leaves the GPU
// idling between bursts. This cache keeps the ggml context / tensors / graph /
// backend buffer alive across calls and reuses them whenever the shape
// signature matches, so a hot call only refreshes Q/K/V/mask data and runs the
// graph. Mirrors PagedAttnSession in ggml_ops_paged_attention.cpp.
//
// kvLen is bucketed to the next power of two so monotonically-growing prefill
// chunks (and the fixed-window SWA layers) hit a small set of cached graphs;
// the K/V padding region [kvLen, bucket) is zeroed (tracked per session) so the
// masked-out tail never feeds NaNs into the softmax→V matmul. Head-first
// (inputFormat == 0) only; flat input falls back to the per-call path.
// ============================================================================
namespace
{
    inline bool prefill_attn_cache_enabled()
    {
        static const bool enabled = []{
            const char* e = std::getenv("TS_PREFILL_ATTN_CACHE");
            return !(e != nullptr && std::strcmp(e, "0") == 0);
        }();
        return enabled;
    }

    // The non-cached (large-kvLen) prefill path streams K/V through
    // ggml_flash_attn_ext instead of materializing the [kvLen, seqLen, numHeads]
    // scores + softmax tensors. That O(N^2) materialization OOMs on long prompts
    // (a 32K-token global window alone needs multiple GB just for scores+probs)
    // and is also the slower, numerically-fragile path for multi-token Q. Flash
    // is on by default; TS_PREFILL_ATTN_FLASH=0 reverts to the materialized graph.
    inline bool prefill_attn_flash_enabled()
    {
        static const bool enabled = []{
            const char* e = std::getenv("TS_PREFILL_ATTN_FLASH");
            return !(e != nullptr && std::strcmp(e, "0") == 0);
        }();
        return enabled;
    }

    inline std::uint32_t prefill_float_bits(float v)
    {
        std::uint32_t b;
        std::memcpy(&b, &v, sizeof(b));
        return b;
    }

    inline int prefill_kv_bucket(int kv_len)
    {
        if (kv_len <= 64) return 64;
        int v = kv_len - 1;
        v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
        return v + 1;
    }

    // Only cache shapes whose retained per-session buffer stays modest. The
    // scores/probs intermediates (~kv_bucket*seq_len*num_heads*4 bytes each) are
    // allocated INTO the retained session buffer, so a large-kvLen session would
    // pin hundreds of MB / GB of VRAM. Those large-bucket calls are compute-bound
    // anyway (per-call graph-build overhead is a small fraction of their flash
    // compute), so leaving them on the per-call path costs little while keeping
    // the cache's VRAM footprint bounded — important near GPU capacity. The big
    // cache win is the many small-bucket, overhead-dominated calls (e.g. the SWA
    // local layers: fixed ~window-sized kvLen, hundreds of calls per forward).
    inline bool prefill_should_cache(int kv_bucket, int seq_len, int num_heads)
    {
        const long long scoresBytes = static_cast<long long>(kv_bucket) * seq_len * num_heads * 4;
        return scoresBytes <= (192LL << 20); // 192 MiB per scores tensor
    }

    struct PrefillAttnSession
    {
        bool valid = false;
        int num_q = 0;
        int kv_bucket = 0;
        int num_heads = 0;
        int num_kv_heads = 0;
        int head_dim = 0;
        std::uint32_t scale_bits = 0;
        bool kv_f16 = false;

        std::unique_ptr<unsigned char[]> ctx_mem;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;

        ggml_tensor* q_in = nullptr;
        ggml_tensor* k_in = nullptr;
        ggml_tensor* v_in = nullptr;
        ggml_tensor* mask = nullptr;
        ggml_tensor* result = nullptr;
        ggml_cgraph* graph = nullptr;

        std::uint64_t lru = 0;
        int kv_zero_covered_from = 0; // max kvLen written into the K/V padding region

        void destroy()
        {
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            ctx_mem.reset();
            q_in = k_in = v_in = mask = result = nullptr;
            graph = nullptr;
            valid = false;
            kv_zero_covered_from = 0;
        }

        ~PrefillAttnSession() { destroy(); }
        PrefillAttnSession() = default;
        PrefillAttnSession(const PrefillAttnSession&) = delete;
        PrefillAttnSession& operator=(const PrefillAttnSession&) = delete;
    };

    constexpr std::size_t kPrefillAttnCacheSize = 16;
    thread_local std::array<PrefillAttnSession, kPrefillAttnCacheSize> g_prefill_attn_cache;
    thread_local std::uint64_t g_prefill_attn_lru = 0;
    thread_local std::vector<ggml_fp16_t> g_prefill_mask_scratch;
    thread_local std::vector<unsigned char> g_prefill_zero_scratch;

    PrefillAttnSession* find_prefill_session(
        int num_q, int kv_bucket, int num_heads, int num_kv_heads, int head_dim,
        std::uint32_t scale_bits, bool kv_f16)
    {
        for (auto& s : g_prefill_attn_cache)
        {
            if (s.valid && s.num_q == num_q && s.kv_bucket == kv_bucket &&
                s.num_heads == num_heads && s.num_kv_heads == num_kv_heads &&
                s.head_dim == head_dim && s.scale_bits == scale_bits && s.kv_f16 == kv_f16)
            {
                s.lru = ++g_prefill_attn_lru;
                return &s;
            }
        }
        return nullptr;
    }

    PrefillAttnSession& acquire_prefill_session_slot()
    {
        for (auto& s : g_prefill_attn_cache)
            if (!s.valid) { s.lru = ++g_prefill_attn_lru; return s; }
        PrefillAttnSession* victim = &g_prefill_attn_cache[0];
        for (auto& s : g_prefill_attn_cache)
            if (s.lru < victim->lru) victim = &s;
        victim->destroy();
        victim->lru = ++g_prefill_attn_lru;
        return *victim;
    }

    bool build_prefill_session(
        PrefillAttnSession& s,
        int num_q, int kv_bucket, int num_heads, int num_kv_heads, int head_dim,
        float scale, bool kv_f16)
    {
        constexpr std::size_t kCtxMemSize = 1024 * 1024;
        s.ctx_mem.reset(new (std::nothrow) unsigned char[kCtxMemSize]);
        if (!s.ctx_mem) { set_last_error("prefill-attn session: alloc ctx mem failed."); return false; }

        ggml_init_params params = {};
        params.mem_size = kCtxMemSize;
        params.mem_buffer = s.ctx_mem.get();
        params.no_alloc = true;
        s.ctx = ggml_init(params);
        if (s.ctx == nullptr) { set_last_error("prefill-attn session: ggml_init failed."); s.ctx_mem.reset(); return false; }

        const ggml_type kv_type = kv_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        // Head-first: GGML [headDim, seq, heads].
        s.q_in = ggml_new_tensor_3d(s.ctx, GGML_TYPE_F32, head_dim, num_q, num_heads);
        s.k_in = ggml_new_tensor_3d(s.ctx, kv_type, head_dim, kv_bucket, num_kv_heads);
        s.v_in = ggml_new_tensor_3d(s.ctx, kv_type, head_dim, kv_bucket, num_kv_heads);
        s.mask = ggml_new_tensor_4d(s.ctx, GGML_TYPE_F16, kv_bucket, num_q, 1, 1);
        s.result = ggml_new_tensor_2d(s.ctx, GGML_TYPE_F32, num_heads * head_dim, num_q);
        if (!s.q_in || !s.k_in || !s.v_in || !s.mask || !s.result)
        {
            set_last_error("prefill-attn session: tensor alloc failed.");
            ggml_free(s.ctx); s.ctx = nullptr; s.ctx_mem.reset(); return false;
        }

        // scores = mul_mat(K, Q) with GQA broadcast; F32 accumulation.
        ggml_tensor* scores = ggml_mul_mat(s.ctx, s.k_in, s.q_in);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        ggml_tensor* probs = ggml_soft_max_ext(s.ctx, scores, s.mask, scale, 0.0f);
        ggml_tensor* v_perm = ggml_cont(s.ctx, ggml_permute(s.ctx, s.v_in, 1, 0, 2, 3));
        ggml_tensor* attn_out = ggml_mul_mat(s.ctx, v_perm, probs);
        ggml_tensor* attn_perm = ggml_permute(s.ctx, attn_out, 0, 2, 1, 3);
        ggml_tensor* attn_cont = ggml_cont(s.ctx, attn_perm);
        ggml_tensor* attn_flat = ggml_reshape_2d(s.ctx, attn_cont, num_heads * head_dim, num_q);
        ggml_tensor* out = ggml_cpy(s.ctx, attn_flat, s.result);
        ggml_set_output(out);

        s.graph = ggml_new_graph(s.ctx);
        ggml_build_forward_expand(s.graph, out);

        s.buffer = ggml_backend_alloc_ctx_tensors(s.ctx, g_backend);
        if (s.buffer == nullptr)
        {
            set_last_error("prefill-attn session: backend buffer alloc failed.");
            ggml_free(s.ctx); s.ctx = nullptr; s.ctx_mem.reset(); return false;
        }

        s.num_q = num_q; s.kv_bucket = kv_bucket; s.num_heads = num_heads;
        s.num_kv_heads = num_kv_heads; s.head_dim = head_dim;
        s.scale_bits = prefill_float_bits(scale); s.kv_f16 = kv_f16;
        s.kv_zero_covered_from = 0; // backend buffer starts zero-initialised
        s.valid = true;
        return true;
    }

    // Cached head-first fused prefill attention. q_data is F32 [numHeads, seqLen,
    // headDim]; k/v_data are F32 [numKVHeads, kvStride, headDim] or F16 of the same
    // layout (kvStride = the source seq stride, e.g. the F16 cache's cacheLen, of
    // which the leading kvLen rows are read). out_data is F32 [seqLen, numHeads*headDim].
    int fused_prefill_attn_cached(
        const void* q_data, const void* k_data, const void* v_data, float* out_data,
        int num_heads, int num_kv_heads, int head_dim,
        int seq_len, int kv_len, int kv_stride,
        int mask_start_pos, int sliding_window, float scale, bool kv_f16)
    {
        if (!ensure_backend()) return 0;

        const int kv_bucket = prefill_kv_bucket(kv_len);
        const std::uint32_t scale_bits = prefill_float_bits(scale);

        PrefillAttnSession* sess = find_prefill_session(
            seq_len, kv_bucket, num_heads, num_kv_heads, head_dim, scale_bits, kv_f16);
        if (sess == nullptr)
        {
            PrefillAttnSession& slot = acquire_prefill_session_slot();
            if (!build_prefill_session(slot, seq_len, kv_bucket, num_heads, num_kv_heads,
                                       head_dim, scale, kv_f16))
                return 0;
            sess = &slot;
        }

        const std::size_t kv_elem = kv_f16 ? sizeof(ggml_fp16_t) : sizeof(float);
        const std::size_t q_bytes = static_cast<std::size_t>(num_heads) * seq_len * head_dim * sizeof(float);

        // The previous graph_compute on this cached session may still be reading
        // q_in/k_in/v_in; drain before overwriting them.
        host_read_barrier();

        ggml_backend_tensor_set(sess->q_in, q_data, 0, q_bytes);

        // Upload K/V: leading kv_len rows per head into the bucket-sized tensor.
        const std::size_t srcHeadElems = static_cast<std::size_t>(kv_stride) * head_dim;
        const std::size_t dstHeadElems = static_cast<std::size_t>(kv_bucket) * head_dim;
        const std::size_t rowBytes = static_cast<std::size_t>(kv_len) * head_dim * kv_elem;
        const auto* kb = static_cast<const unsigned char*>(k_data);
        const auto* vb = static_cast<const unsigned char*>(v_data);
        for (int h = 0; h < num_kv_heads; ++h)
        {
            const std::size_t srcOff = static_cast<std::size_t>(h) * srcHeadElems * kv_elem;
            const std::size_t dstOff = static_cast<std::size_t>(h) * dstHeadElems * kv_elem;
            ggml_backend_tensor_set(sess->k_in, kb + srcOff, dstOff, rowBytes);
            ggml_backend_tensor_set(sess->v_in, vb + srcOff, dstOff, rowBytes);
        }

        // Zero the K/V padding [kv_len, covered) per head only when this call is
        // shorter than a previous one (otherwise the tail is still zero from the
        // freshly-allocated buffer / never written).
        if (kv_len < sess->kv_zero_covered_from)
        {
            const int zero_rows = sess->kv_zero_covered_from - kv_len;
            const std::size_t zeroBytes = static_cast<std::size_t>(zero_rows) * head_dim * kv_elem;
            if (g_prefill_zero_scratch.size() < zeroBytes)
                g_prefill_zero_scratch.assign(zeroBytes, 0);
            for (int h = 0; h < num_kv_heads; ++h)
            {
                const std::size_t padOff =
                    (static_cast<std::size_t>(h) * dstHeadElems + static_cast<std::size_t>(kv_len) * head_dim) * kv_elem;
                ggml_backend_tensor_set(sess->k_in, g_prefill_zero_scratch.data(), padOff, zeroBytes);
                ggml_backend_tensor_set(sess->v_in, g_prefill_zero_scratch.data(), padOff, zeroBytes);
            }
        }
        if (kv_len > sess->kv_zero_covered_from)
            sess->kv_zero_covered_from = kv_len;

        // Build + upload the causal (+ optional sliding window) mask over the full
        // bucket. threshold < kv_len, so the [kv_len, bucket) tail is auto-masked.
        g_prefill_mask_scratch.assign(static_cast<std::size_t>(kv_bucket) * seq_len, ggml_fp32_to_fp16(0.0f));
        {
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
            for (int q_idx = 0; q_idx < seq_len; q_idx++)
            {
                int threshold = mask_start_pos + q_idx;
                int winStart = (sliding_window > 0) ? std::max(0, threshold - sliding_window + 1) : 0;
                ggml_fp16_t* row = &g_prefill_mask_scratch[static_cast<std::size_t>(q_idx) * kv_bucket];
                for (int kv_idx = 0; kv_idx < kv_bucket; kv_idx++)
                    row[kv_idx] = (kv_idx > threshold || kv_idx < winStart) ? neg_inf : zero_val;
            }
        }
        ggml_backend_tensor_set(sess->mask, g_prefill_mask_scratch.data(), 0,
                                 g_prefill_mask_scratch.size() * sizeof(ggml_fp16_t));

        ggml_status status = ggml_backend_graph_compute(g_backend, sess->graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml graph compute failed for cached fused prefill attention.");
            return 0;
        }

        finalize_compute_with_download(sess->result, out_data, q_bytes);
        clear_last_error();
        return 1;
    }

    // Flash-attention prefill: same contract as the materialized scores path in
    // the F16KV / F32 exports below, but runs ggml_flash_attn_ext, which streams
    // the K/V tiles instead of materializing the [kvLen, seqLen, numHeads] score
    // and softmax intermediates. Those O(N^2) buffers are what OOMs on long
    // prompts (a single 32K-token global window needs GBs for scores+probs) and
    // are the slower, F16-accumulator-fragile path for multi-token Q. Flash keeps
    // the attention working set to O(N) and matches the dense/MoE verify prefill.
    //
    // q_data: F32 head-first [numHeads, seqLen, headDim] == GGML [headDim, seqLen, numHeads].
    // k/v_data: [numKVHeads, kvStride, headDim] F16 or F32; leading kvLen rows read per head.
    // out_data: F32 flat [seqLen, numHeads*headDim] == GGML [numHeads*headDim, seqLen].
    // GQA (numHeads % numKVHeads == 0) is handled in-kernel. Returns 1 on success,
    // 0 on hard failure, -1 when the backend has no flash kernel for this shape
    // (caller falls back to the materialized graph).
    int fused_prefill_attn_flash(
        const void* q_data, const void* k_data, const void* v_data, float* out_data,
        int num_heads, int num_kv_heads, int head_dim,
        int seq_len, int kv_len, int kv_stride,
        int mask_start_pos, int sliding_window, float scale, bool kv_f16)
    {
        if (!ensure_backend()) return 0;

        const std::size_t ctx_size = 4 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context for flash prefill attention.");
            return 0;
        }
        auto* ctx = context.value;

        const ggml_type kv_type = kv_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        // flash_attn_ext wants q=[headDim, seqLen, numHeads], k/v=[headDim, kvLen,
        // numKVHeads] — exactly the head-first layout, so no permutes are needed.
        ggml_tensor* q_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, seq_len, num_heads);
        ggml_tensor* k_in = ggml_new_tensor_3d(ctx, kv_type, head_dim, kv_len, num_kv_heads);
        ggml_tensor* v_in = ggml_new_tensor_3d(ctx, kv_type, head_dim, kv_len, num_kv_heads);
        ggml_tensor* mask_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_len, seq_len);
        ggml_tensor* attn_result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, num_heads * head_dim, seq_len);
        if (q_in == nullptr || k_in == nullptr || v_in == nullptr ||
            mask_tensor == nullptr || attn_result == nullptr)
        {
            set_last_error("Failed to create ggml tensors for flash prefill attention.");
            return 0;
        }

        ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q_in, k_in, v_in, mask_tensor, scale, 0.0f, 0.0f);
        if (attn_out == nullptr) { set_last_error("Failed flash_attn_ext node."); return 0; }
        ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
        if (!backend_supports_op(attn_out))
            return -1; // no flash kernel for this geometry; let the caller fall back

        // flash_attn_ext returns [headDim, numHeads, seqLen, 1] — contiguous, with
        // headDim innermost then head index, exactly the flat [numHeads*headDim, seqLen] output.
        ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_out, num_heads * head_dim, seq_len);
        ggml_tensor* output = ggml_cpy(ctx, attn_flat, attn_result);
        if (output == nullptr) { set_last_error("Failed flash output cpy node."); return 0; }
        ggml_set_output(output);

        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate backend buffer for flash prefill attention.");
            return 0;
        }

        host_read_barrier();
        ggml_backend_tensor_set(q_in, q_data, 0,
            static_cast<std::size_t>(num_heads) * seq_len * head_dim * sizeof(float));

        // Upload the leading kvLen rows of each head from the [numKVHeads, kvStride,
        // headDim] cache. Contiguous single upload when the cache is exactly sized.
        const std::size_t kv_elem = kv_f16 ? sizeof(ggml_fp16_t) : sizeof(float);
        const std::size_t dstHeadElems = static_cast<std::size_t>(kv_len) * head_dim;
        if (kv_stride == kv_len)
        {
            const std::size_t bytes = static_cast<std::size_t>(num_kv_heads) * dstHeadElems * kv_elem;
            ggml_backend_tensor_set(k_in, k_data, 0, bytes);
            ggml_backend_tensor_set(v_in, v_data, 0, bytes);
        }
        else
        {
            const auto* kb = static_cast<const unsigned char*>(k_data);
            const auto* vb = static_cast<const unsigned char*>(v_data);
            const std::size_t srcHeadElems = static_cast<std::size_t>(kv_stride) * head_dim;
            const std::size_t headBytes = dstHeadElems * kv_elem;
            for (int h = 0; h < num_kv_heads; ++h)
            {
                const std::size_t srcOff = static_cast<std::size_t>(h) * srcHeadElems * kv_elem;
                const std::size_t dstOff = static_cast<std::size_t>(h) * dstHeadElems * kv_elem;
                ggml_backend_tensor_set(k_in, kb + srcOff, dstOff, headBytes);
                ggml_backend_tensor_set(v_in, vb + srcOff, dstOff, headBytes);
            }
        }

        // Causal (+ optional sliding-window) additive mask: row q attends key k iff
        // winStart <= k <= mask_start_pos + q.
        g_prefill_mask_scratch.assign(static_cast<std::size_t>(kv_len) * seq_len, ggml_fp32_to_fp16(0.0f));
        {
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
            for (int q_idx = 0; q_idx < seq_len; q_idx++)
            {
                int threshold = mask_start_pos + q_idx;
                int winStart = (sliding_window > 0) ? std::max(0, threshold - sliding_window + 1) : 0;
                ggml_fp16_t* row = &g_prefill_mask_scratch[static_cast<std::size_t>(q_idx) * kv_len];
                for (int kv_idx = 0; kv_idx < kv_len; kv_idx++)
                    row[kv_idx] = (kv_idx > threshold || kv_idx < winStart) ? neg_inf : zero_val;
            }
        }
        ggml_backend_tensor_set(mask_tensor, g_prefill_mask_scratch.data(), 0,
                                 g_prefill_mask_scratch.size() * sizeof(ggml_fp16_t));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml graph compute failed for flash prefill attention.");
            return 0;
        }

        finalize_compute_with_download(attn_result, out_data,
            static_cast<std::size_t>(num_heads) * seq_len * head_dim * sizeof(float));
        clear_last_error();
        return 1;
    }
} // anonymous namespace

// ============================================================================
// Fused prefill attention: Q*K^T → causal mask → softmax → *V as one GGML
// graph. Eliminates ~5 separate C# → GGML round trips per layer.
//
// C# head-first layout [numHeads, seqLen, headDim] maps naturally to GGML
// [headDim, seqLen, numHeads] with no permutation.
//
// Supports GQA (numHeads != numKVHeads) and optional sliding window.
// ============================================================================
TSG_EXPORT int TSGgml_FusedPrefillAttentionF32(
    const float* q_data,      // head-first: [numHeads * seqLen * headDim]  OR  flat: [seqLen * numHeads * headDim]
    const float* k_data,      // head-first: [numKVHeads * kvLen * headDim] OR  flat: [kvLen * numKVHeads * headDim]
    const float* v_data,      // same as k_data layout
    float* out_data,          // always flat: [seqLen * numHeads * headDim]
    int numHeads,
    int numKVHeads,
    int headDim,
    int seqLen,
    int kvLen,
    int maskStartPos,
    int slidingWindow,
    float scale,
    int inputFormat)   // 0 = head-first [numHeads, seqLen, headDim], 1 = flat [seqLen, numHeads*headDim]
{
    try
    {
        if (!ensure_backend())
        {
            return 0;
        }

        // Session-cached fast path (head-first only): reuse the graph + backend
        // buffer across the hundreds of per-layer/per-chunk prefill calls.
        if (inputFormat == 0 && prefill_attn_cache_enabled()
            && prefill_should_cache(prefill_kv_bucket(kvLen), seqLen, numHeads))
        {
            return fused_prefill_attn_cached(
                q_data, k_data, v_data, out_data,
                numHeads, numKVHeads, headDim, seqLen, kvLen, /*kv_stride*/ kvLen,
                maskStartPos, slidingWindow, scale, /*kv_f16*/ false);
        }

        // Large-kvLen head-first path: flash attention streams K/V instead of
        // materializing the O(N^2) scores+softmax that OOMs on long prompts. The
        // flat (inputFormat==1) layout stays on the materialized graph below.
        if (inputFormat == 0 && prefill_attn_flash_enabled())
        {
            int fr = fused_prefill_attn_flash(
                q_data, k_data, v_data, out_data,
                numHeads, numKVHeads, headDim, seqLen, kvLen, /*kv_stride*/ kvLen,
                maskStartPos, slidingWindow, scale, /*kv_f16*/ false);
            if (fr >= 0) return fr;
        }

        const std::size_t ctx_size = 4 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context for fused prefill attention.");
            return 0;
        }

        auto* ctx = context.value;
        const int qSize = numHeads * seqLen * headDim;
        const int kvSize = numKVHeads * kvLen * headDim;

        // Create input tensors matching the C# memory layout
        ggml_tensor* q_in;
        ggml_tensor* k_in;
        ggml_tensor* v_in;
        if (inputFormat == 1)
        {
            // Flat layout: C# [seqLen, numHeads*headDim] == GGML [numHeads*headDim, seqLen]
            q_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, numHeads * headDim, seqLen);
            k_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, numKVHeads * headDim, kvLen);
            v_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, numKVHeads * headDim, kvLen);
        }
        else
        {
            // Head-first: C# [numHeads, seqLen, headDim] == GGML [headDim, seqLen, numHeads]
            q_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, seqLen, numHeads);
            k_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, kvLen, numKVHeads);
            v_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, kvLen, numKVHeads);
        }
        // Output in flat layout [numHeads*headDim, seqLen] (GGML) = [seqLen, numHeads*headDim] (C#)
        // The permute+cont inside the graph handles the transpose from head-first to flat.
        ggml_tensor* attn_result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, numHeads * headDim, seqLen);

        // Build causal + optional sliding window mask in FP16: [kvLen, seqLen]
        ggml_tensor* mask_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kvLen, seqLen, 1, 1);
        std::vector<ggml_fp16_t> mask_data(static_cast<std::size_t>(kvLen) * seqLen);
        {
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
            for (int q_idx = 0; q_idx < seqLen; q_idx++)
            {
                int threshold = maskStartPos + q_idx;
                int winStart = (slidingWindow > 0) ? std::max(0, threshold - slidingWindow + 1) : 0;
                ggml_fp16_t* row = &mask_data[static_cast<std::size_t>(q_idx) * kvLen];
                for (int kv_idx = 0; kv_idx < kvLen; kv_idx++)
                    row[kv_idx] = (kv_idx > threshold || kv_idx < winStart) ? neg_inf : zero_val;
            }
        }

        if (q_in == nullptr || k_in == nullptr || v_in == nullptr || attn_result == nullptr || mask_tensor == nullptr)
        {
            set_last_error("Failed to create ggml tensors for fused prefill attention.");
            return 0;
        }

        // For flat input format, reshape [dim, seqLen] → [headDim, numHeads, seqLen]
        // then permute to [headDim, seqLen, numHeads]. These are free graph ops (no data copy).
        ggml_tensor* q_attn = q_in;
        ggml_tensor* k_attn = k_in;
        ggml_tensor* v_attn = v_in;
        if (inputFormat == 1)
        {
            // Q: [numHeads*headDim, seqLen] → [headDim, numHeads, seqLen] → permute(0,2,1,3) → [headDim, seqLen, numHeads]
            ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_in, headDim, numHeads, seqLen);
            q_attn = ggml_permute(ctx, q_3d, 0, 2, 1, 3);
            q_attn = ggml_cont(ctx, q_attn);
            // K: [numKVHeads*headDim, kvLen] → [headDim, numKVHeads, kvLen] → permute → [headDim, kvLen, numKVHeads]
            ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_in, headDim, numKVHeads, kvLen);
            k_attn = ggml_permute(ctx, k_3d, 0, 2, 1, 3);
            k_attn = ggml_cont(ctx, k_attn);
            // V: same as K
            ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_in, headDim, numKVHeads, kvLen);
            v_attn = ggml_permute(ctx, v_3d, 0, 2, 1, 3);
            v_attn = ggml_cont(ctx, v_attn);
        }

        // mul_mat(K, Q): K=[headDim, kvLen, numKVHeads], Q=[headDim, seqLen, numHeads]
        // GQA broadcast: numHeads must be a multiple of numKVHeads.
        ggml_tensor* scores = ggml_mul_mat(ctx, k_attn, q_attn);
        if (scores == nullptr)
        {
            set_last_error("Failed to create Q*K^T matmul node.");
            return 0;
        }
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);

        // Softmax with mask: softmax(scores * scale + mask)
        ggml_tensor* probs = ggml_soft_max_ext(ctx, scores, mask_tensor, scale, 0.0f);
        if (probs == nullptr)
        {
            set_last_error("Failed to create softmax node.");
            return 0;
        }

        // V permute for value matmul: [headDim, kvLen, numKVHeads] → [kvLen, headDim, numKVHeads]
        ggml_tensor* v_perm = ggml_permute(ctx, v_attn, 1, 0, 2, 3);
        v_perm = ggml_cont(ctx, v_perm);

        // attn = probs * V_perm → [headDim, seqLen, numHeads]
        ggml_tensor* attn_out = ggml_mul_mat(ctx, v_perm, probs);
        if (attn_out == nullptr)
        {
            set_last_error("Failed to create scores*V matmul node.");
            return 0;
        }

        // Permute to flat layout [headDim*numHeads, seqLen] (GGML)
        // = [seqLen, numHeads*headDim] (C#), skipping the C# ReshapeFromHeadsEx copy.
        // [headDim, seqLen, numHeads] → permute(0,2,1,3) → [headDim, numHeads, seqLen]
        // → cont → reshape to [headDim*numHeads, seqLen]
        ggml_tensor* attn_perm = ggml_permute(ctx, attn_out, 0, 2, 1, 3);
        ggml_tensor* attn_cont = ggml_cont(ctx, attn_perm);
        ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_cont, numHeads * headDim, seqLen);

        // Copy result to download target
        ggml_tensor* output = ggml_cpy(ctx, attn_flat, attn_result);
        if (output == nullptr)
        {
            set_last_error("Failed to create output copy node.");
            return 0;
        }
        ggml_set_output(output);

        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate backend buffer for fused prefill attention.");
            return 0;
        }

        // Upload input data. q/k/v_data are C# tensor pointers that may have
        // pending GPU writes from a previous async op (e.g. zero-copy QKV matmul);
        // drain first so the CPU-side memcpy underneath ggml_backend_tensor_set
        // doesn't race with in-flight writes.
        host_read_barrier();
        ggml_backend_tensor_set(q_in, q_data, 0, qSize * sizeof(float));
        ggml_backend_tensor_set(k_in, k_data, 0, kvSize * sizeof(float));
        ggml_backend_tensor_set(v_in, v_data, 0, kvSize * sizeof(float));
        ggml_backend_tensor_set(mask_tensor, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml graph compute failed for fused prefill attention.");
            return 0;
        }

        finalize_compute_with_download(attn_result, out_data, qSize * sizeof(float));

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
        set_last_error("Unknown fused prefill attention failure.");
        return 0;
    }
}

// ============================================================================
// Fused prefill attention reading the K/V straight out of an F16 cache, with NO
// host F16->F32 dequant. Q stays F32 (head-first [numHeads, seqLen, headDim]);
// K/V are uploaded as F16 directly from the persistent cache
// ([numKVHeads, kvCacheLen, headDim] head-first), reading the leading kvLen rows
// (kvCacheLen is the allocated stride, which can exceed kvLen). mul_mat with the
// F16 K/V and GGML_PREC_F32 accumulation is numerically identical to dequantizing
// to F32 first, so this is a drop-in replacement for the
// ExpandKVHeads(F16->F32) + FusedPrefillAttention(F32) chain — it just removes
// the per-chunk dequant round-trip (and halves the K/V upload bytes).
//
// Same fused mul_mat -> soft_max_ext -> mul_mat graph as the F32 path; GQA is
// handled by ggml's mul_mat broadcast (numHeads % numKVHeads == 0).
// ============================================================================
TSG_EXPORT int TSGgml_FusedPrefillAttentionF16KV(
    const float* q_data,        // head-first F32 [numHeads, seqLen, headDim]
    const void* k_data,         // head-first F16 cache [numKVHeads, kvCacheLen, headDim]
    const void* v_data,         // same layout
    float* out_data,            // flat F32 [seqLen, numHeads*headDim]
    int numHeads,
    int numKVHeads,
    int headDim,
    int seqLen,
    int kvLen,
    int kvCacheLen,             // allocated seq stride of the cache (>= kvLen)
    int maskStartPos,
    int slidingWindow,
    float scale)
{
    try
    {
        if (!ensure_backend())
            return 0;
        if (numHeads <= 0 || numKVHeads <= 0 || headDim <= 0 || seqLen <= 0 || kvLen <= 0 ||
            kvCacheLen < kvLen || numHeads % numKVHeads != 0)
        {
            set_last_error("Invalid dimensions for fused prefill attention (F16 KV).");
            return 0;
        }

        // Session-cached fast path: reuse the graph + backend buffer across calls.
        if (prefill_attn_cache_enabled()
            && prefill_should_cache(prefill_kv_bucket(kvLen), seqLen, numHeads))
        {
            return fused_prefill_attn_cached(
                q_data, k_data, v_data, out_data,
                numHeads, numKVHeads, headDim, seqLen, kvLen, /*kv_stride*/ kvCacheLen,
                maskStartPos, slidingWindow, scale, /*kv_f16*/ true);
        }

        // Large-kvLen path: flash attention streams K/V instead of materializing
        // the O(N^2) scores+softmax that OOMs on long prompts. Falls through to the
        // materialized graph below only if the backend has no flash kernel here.
        if (prefill_attn_flash_enabled())
        {
            int fr = fused_prefill_attn_flash(
                q_data, k_data, v_data, out_data,
                numHeads, numKVHeads, headDim, seqLen, kvLen, /*kv_stride*/ kvCacheLen,
                maskStartPos, slidingWindow, scale, /*kv_f16*/ true);
            if (fr >= 0) return fr;
        }

        const std::size_t ctx_size = 4 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context for fused prefill attention (F16 KV).");
            return 0;
        }
        auto* ctx = context.value;

        // Head-first layout: C# [numHeads, seqLen, headDim] == GGML [headDim, seqLen, numHeads].
        ggml_tensor* q_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, seqLen, numHeads);
        ggml_tensor* k_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, headDim, kvLen, numKVHeads);
        ggml_tensor* v_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, headDim, kvLen, numKVHeads);
        ggml_tensor* attn_result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, numHeads * headDim, seqLen);
        ggml_tensor* mask_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kvLen, seqLen, 1, 1);

        if (q_in == nullptr || k_in == nullptr || v_in == nullptr ||
            attn_result == nullptr || mask_tensor == nullptr)
        {
            set_last_error("Failed to create ggml tensors for fused prefill attention (F16 KV).");
            return 0;
        }

        std::vector<ggml_fp16_t> mask_data(static_cast<std::size_t>(kvLen) * seqLen);
        {
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
            for (int q_idx = 0; q_idx < seqLen; q_idx++)
            {
                int threshold = maskStartPos + q_idx;
                int winStart = (slidingWindow > 0) ? std::max(0, threshold - slidingWindow + 1) : 0;
                ggml_fp16_t* row = &mask_data[static_cast<std::size_t>(q_idx) * kvLen];
                for (int kv_idx = 0; kv_idx < kvLen; kv_idx++)
                    row[kv_idx] = (kv_idx > threshold || kv_idx < winStart) ? neg_inf : zero_val;
            }
        }

        // scores = mul_mat(K_f16, Q_f32); GQA broadcast over heads. F32 accumulation.
        ggml_tensor* scores = ggml_mul_mat(ctx, k_in, q_in);
        if (scores == nullptr) { set_last_error("Failed Q*K^T (F16 KV)."); return 0; }
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);

        ggml_tensor* probs = ggml_soft_max_ext(ctx, scores, mask_tensor, scale, 0.0f);
        if (probs == nullptr) { set_last_error("Failed softmax (F16 KV)."); return 0; }

        // V permute [headDim, kvLen, numKVHeads] -> [kvLen, headDim, numKVHeads] (stays F16).
        ggml_tensor* v_perm = ggml_cont(ctx, ggml_permute(ctx, v_in, 1, 0, 2, 3));
        ggml_tensor* attn_out = ggml_mul_mat(ctx, v_perm, probs);
        if (attn_out == nullptr) { set_last_error("Failed scores*V (F16 KV)."); return 0; }

        // [headDim, seqLen, numHeads] -> flat [headDim*numHeads, seqLen].
        ggml_tensor* attn_perm = ggml_permute(ctx, attn_out, 0, 2, 1, 3);
        ggml_tensor* attn_cont = ggml_cont(ctx, attn_perm);
        ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_cont, numHeads * headDim, seqLen);
        ggml_tensor* output = ggml_cpy(ctx, attn_flat, attn_result);
        if (output == nullptr) { set_last_error("Failed output cpy (F16 KV)."); return 0; }
        ggml_set_output(output);

        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate backend buffer for fused prefill attention (F16 KV).");
            return 0;
        }

        host_read_barrier();
        ggml_backend_tensor_set(q_in, q_data, 0,
            static_cast<std::size_t>(numHeads) * seqLen * headDim * sizeof(float));

        // Upload K/V F16 directly from the cache, reading the leading kvLen rows
        // of each head. When the cache is exactly sized (kvCacheLen == kvLen) the
        // slice is contiguous and a single upload suffices; otherwise upload
        // per-head to skip the unused [kvLen, kvCacheLen) tail.
        const std::size_t dstHeadElems = static_cast<std::size_t>(kvLen) * headDim;
        if (kvCacheLen == kvLen)
        {
            const std::size_t bytes = static_cast<std::size_t>(numKVHeads) * dstHeadElems * sizeof(ggml_fp16_t);
            ggml_backend_tensor_set(k_in, k_data, 0, bytes);
            ggml_backend_tensor_set(v_in, v_data, 0, bytes);
        }
        else
        {
            const ggml_fp16_t* ksrc = static_cast<const ggml_fp16_t*>(k_data);
            const ggml_fp16_t* vsrc = static_cast<const ggml_fp16_t*>(v_data);
            const std::size_t srcHeadElems = static_cast<std::size_t>(kvCacheLen) * headDim;
            const std::size_t headBytes = dstHeadElems * sizeof(ggml_fp16_t);
            for (int h = 0; h < numKVHeads; ++h)
            {
                const std::size_t dstOff = static_cast<std::size_t>(h) * dstHeadElems * sizeof(ggml_fp16_t);
                ggml_backend_tensor_set(k_in, ksrc + static_cast<std::size_t>(h) * srcHeadElems, dstOff, headBytes);
                ggml_backend_tensor_set(v_in, vsrc + static_cast<std::size_t>(h) * srcHeadElems, dstOff, headBytes);
            }
        }
        ggml_backend_tensor_set(mask_tensor, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml graph compute failed for fused prefill attention (F16 KV).");
            return 0;
        }

        finalize_compute_with_download(attn_result, out_data,
            static_cast<std::size_t>(numHeads) * seqLen * headDim * sizeof(float));

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
        set_last_error("Unknown fused prefill attention (F16 KV) failure.");
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

// In-place softmax with causal+SWA mask and optional attention sinks.
// See attention_softmax_with_sinks_f32_impl above for the full contract.
// Replaces the GptOss CPU softmax-with-sinks loop, which was the dominant
// prefill cost on that model.
TSG_EXPORT int TSGgml_AttentionSoftmaxWithSinksF32(
    TensorView3DDesc scores,
    const float* sinks_data,
    int num_heads,
    int seq_len,
    int kv_len,
    int mask_start_pos,
    int sliding_window,
    float scale)
{
    try
    {
        return attention_softmax_with_sinks_f32_impl(
            scores, sinks_data, num_heads, seq_len, kv_len,
            mask_start_pos, sliding_window, scale);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml attention softmax with sinks failure.");
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
