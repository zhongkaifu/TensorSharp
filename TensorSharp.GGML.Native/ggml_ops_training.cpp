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
