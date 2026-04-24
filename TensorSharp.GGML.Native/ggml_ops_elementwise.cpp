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

    // Fused activation-multiply on a single contiguous [N, 2H] gate_up tensor.
    // result[i, j] = act(gate_up[i, j]) * gate_up[i, j + H]   for j in [0, H)
    //
    // This avoids the two large NewContiguous copies that the "split + 2 separate
    // tensors" path would require. Both halves are exposed as ggml_view_2d on the
    // same underlying gate_up buffer, with byte_offset 0 and H*sizeof(float).
    int fused_act_mul_split_f32_impl(
        FusedActMulOpCode op,
        const TensorView2DDesc& result_desc,
        const TensorView2DDesc& gate_up_desc,
        int half_dim)
    {
        if (!ensure_backend())
            return 0;

        if (!validate_desc(result_desc, "result") || !validate_desc(gate_up_desc, "gate_up"))
            return 0;

        if (half_dim <= 0)
        {
            set_last_error("fused_act_mul_split: half_dim must be positive.");
            return 0;
        }
        if (result_desc.stride1 != 1 || gate_up_desc.stride1 != 1)
        {
            set_last_error("fused_act_mul_split: tensors must be row-major (stride1 == 1).");
            return 0;
        }
        if (result_desc.dim0 != gate_up_desc.dim0
            || result_desc.dim1 != half_dim
            || gate_up_desc.dim1 != 2 * half_dim)
        {
            set_last_error("fused_act_mul_split: shape mismatch.");
            return 0;
        }
        // The gate_up tensor itself only needs to be row-major; rows can have
        // padding (stride0 >= 2H). The split offsets are within a single row.
        if (gate_up_desc.stride0 < gate_up_desc.dim1)
        {
            set_last_error("fused_act_mul_split: gate_up row stride must be >= dim1.");
            return 0;
        }

        bool use_zero_copy = true;
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 1 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        // Promote the 2D descs to 4D so we can reuse the existing zero-copy helper.
        auto to_4d = [](const TensorView2DDesc& d) {
            TensorView4DDesc out{};
            out.data = d.data;
            out.ne0 = d.dim1;
            out.ne1 = d.dim0;
            out.ne2 = 1;
            out.ne3 = 1;
            out.nb1 = static_cast<std::int64_t>(d.stride0) * static_cast<std::int64_t>(sizeof(float));
            out.nb2 = out.nb1 * d.dim0;
            out.nb3 = out.nb2;
            out.raw_bytes = d.raw_bytes;
            return out;
        };

        TensorView4DDesc result_4d = to_4d(result_desc);
        TensorView4DDesc gate_up_4d = to_4d(gate_up_desc);

        TensorBinding result_binding, gate_up_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_4d, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, gate_up_4d, gate_up_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            // Standard upload path: bind result + a stand-in for the full gate_up.
            result_binding = create_standard_binding(context.value, result_4d);
            gate_up_binding = create_standard_binding(context.value, gate_up_4d);
        }
        if (result_binding.tensor == nullptr || gate_up_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors for fused_act_mul_split.");
            return 0;
        }

        const std::size_t row_bytes = static_cast<std::size_t>(gate_up_desc.stride0) * sizeof(float);
        const std::size_t half_bytes = static_cast<std::size_t>(half_dim) * sizeof(float);

        ggml_tensor* gate_view = ggml_view_2d(context.value, gate_up_binding.tensor,
            half_dim, gate_up_desc.dim0, row_bytes, 0);
        ggml_tensor* up_view = ggml_view_2d(context.value, gate_up_binding.tensor,
            half_dim, gate_up_desc.dim0, row_bytes, half_bytes);
        if (gate_view == nullptr || up_view == nullptr)
        {
            set_last_error("Failed to create gate/up views for fused_act_mul_split.");
            return 0;
        }

        // ggml_cont packs each strided half-row into a contiguous tensor inside the
        // backend graph. This is required for backends (Metal) whose silu kernel only
        // accepts contiguous inputs. The cont stays GPU/CPU-local within the graph so
        // it is far cheaper than the host-side double NewContiguous it replaces.
        ggml_tensor* gate_cont = ggml_cont(context.value, gate_view);
        ggml_tensor* up_cont = ggml_cont(context.value, up_view);
        if (gate_cont == nullptr || up_cont == nullptr)
        {
            set_last_error("Failed to create gate/up cont nodes for fused_act_mul_split.");
            return 0;
        }

        ggml_tensor* value_tensor = make_fused_act_mul_tensor(context.value, op, gate_cont, up_cont);
        if (value_tensor == nullptr)
        {
            if (g_last_error.empty())
                set_last_error("Failed to create ggml fused_act_mul_split node.");
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, value_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml fused_act_mul_split output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph for fused_act_mul_split.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer for fused_act_mul_split.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(gate_up_binding, gate_up_desc.data, gate_up_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_4d))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed for fused_act_mul_split.");
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

} // namespace

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

TSG_EXPORT int TSGgml_FusedActMulSplitF32(
    int op,
    TensorView2DDesc result,
    TensorView2DDesc gate_up,
    int half_dim)
{
    try
    {
        return fused_act_mul_split_f32_impl(static_cast<FusedActMulOpCode>(op), result, gate_up, half_dim);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml fused activation-multiply-split failure.");
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
