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
