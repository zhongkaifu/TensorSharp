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

        if (g_backend_type == BACKEND_TYPE_CUDA)
        {
            const bool needs_chunking =
                logical_bytes(result_desc) > k_ggml_cuda_max_copy_bytes ||
                logical_bytes(m1_desc) > k_ggml_cuda_max_copy_bytes ||
                static_cast<std::size_t>(result_desc.raw_bytes) > k_ggml_cuda_max_copy_bytes ||
                static_cast<std::size_t>(m1_desc.raw_bytes) > k_ggml_cuda_max_copy_bytes ||
                (beta != 0.0f && (
                    logical_bytes(src_desc) > k_ggml_cuda_max_copy_bytes ||
                    static_cast<std::size_t>(src_desc.raw_bytes) > k_ggml_cuda_max_copy_bytes));

            if (needs_chunking)
            {
                int chunk_rows = rows;
                chunk_rows = limit_rows_for_cuda_copy(chunk_rows, result_desc);
                chunk_rows = limit_rows_for_cuda_copy(chunk_rows, m1_desc);
                if (beta != 0.0f)
                {
                    chunk_rows = limit_rows_for_cuda_copy(chunk_rows, src_desc);
                    if (src_desc.dim0 != rows)
                        chunk_rows = (chunk_rows / src_desc.dim0) * src_desc.dim0;
                }

                if (chunk_rows <= 0)
                {
                    set_last_error("GGML CUDA addmm received a row slice larger than the backend copy limit.");
                    return 0;
                }

                if (chunk_rows < rows)
                {
                    for (int row_start = 0; row_start < rows; row_start += chunk_rows)
                    {
                        const int row_count = std::min(chunk_rows, rows - row_start);
                        const TensorView2DDesc result_slice = slice_rows_2d(result_desc, row_start, row_count);
                        const TensorView2DDesc m1_slice = slice_rows_2d(m1_desc, row_start, row_count);
                        const TensorView2DDesc src_slice = beta == 0.0f
                            ? TensorView2DDesc{}
                            : (src_desc.dim0 == rows ? slice_rows_2d(src_desc, row_start, row_count) : src_desc);

                        if (!addmm_f32_impl(result_slice, src_slice, m1_slice, m2_desc, beta, alpha))
                            return 0;
                    }

                    clear_last_error();
                    return 1;
                }
            }
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

        if (g_backend_type == BACKEND_TYPE_CUDA)
        {
            const bool needs_chunking =
                logical_bytes(result_desc) > k_ggml_cuda_max_copy_bytes ||
                logical_bytes(m1_desc) > k_ggml_cuda_max_copy_bytes ||
                static_cast<std::size_t>(result_desc.raw_bytes) > k_ggml_cuda_max_copy_bytes ||
                static_cast<std::size_t>(m1_desc.raw_bytes) > k_ggml_cuda_max_copy_bytes;

            if (needs_chunking)
            {
                int chunk_rows = rows;
                chunk_rows = limit_rows_for_cuda_copy(chunk_rows, result_desc);
                chunk_rows = limit_rows_for_cuda_copy(chunk_rows, m1_desc);

                if (chunk_rows <= 0)
                {
                    set_last_error("GGML CUDA addmm_quant received a row slice larger than the backend copy limit.");
                    return 0;
                }

                if (chunk_rows < rows)
                {
                    for (int row_start = 0; row_start < rows; row_start += chunk_rows)
                    {
                        const int row_count = std::min(chunk_rows, rows - row_start);
                        const TensorView2DDesc result_slice = slice_rows_2d(result_desc, row_start, row_count);
                        const TensorView2DDesc m1_slice = slice_rows_2d(m1_desc, row_start, row_count);

                        if (!addmm_quant_f32_impl(result_slice, m1_slice, m2_quant))
                            return 0;
                    }

                    clear_last_error();
                    return 1;
                }
            }
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

        // Result/input bindings. If zero-copy host binding fails for either tensor,
        // fall back to standard ggml-owned buffers for both.
        TensorBinding result_binding;
        TensorBinding m1_binding;
        std::vector<float> packed_m1;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t result_buf = nullptr;
            ggml_backend_buffer_t m1_buf = nullptr;
            const bool result_ok = create_binding_from_host_ptr_2d(context.value, g_backend, result_desc, result_binding, result_buf);
            const bool m1_ok = result_ok && create_binding_from_host_ptr_2d(context.value, g_backend, m1_desc, m1_binding, m1_buf);

            if (result_ok && m1_ok)
            {
                host_ptr_buffers.emplace_back(result_buf);
                host_ptr_buffers.emplace_back(m1_buf);
            }
            else
            {
                if (m1_buf != nullptr)
                    ggml_backend_buffer_free(m1_buf);
                if (result_buf != nullptr)
                    ggml_backend_buffer_free(result_buf);

                use_zero_copy = false;
                result_binding = create_standard_binding(context.value, result_desc);
                m1_binding = can_map_standard_view(m1_desc)
                    ? create_standard_binding(context.value, m1_desc)
                    : create_packed_standard_binding(context.value, m1_desc, packed_m1);
            }
        }
        else
        {
            result_binding = create_standard_binding(context.value, result_desc);
            m1_binding = can_map_standard_view(m1_desc)
                ? create_standard_binding(context.value, m1_desc)
                : create_packed_standard_binding(context.value, m1_desc, packed_m1);
        }

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
        bool m2_needs_upload = false;
        {
            ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
            if (dev != nullptr && m2_quant.raw_bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                if (try_get_cacheable_tensor_buffer(
                        g_backend,
                        dev,
                        m2_tensor,
                        m2_quant.data,
                        static_cast<std::size_t>(m2_quant.raw_bytes),
                        buf,
                        addr,
                        m2_needs_upload))
                {
                    ggml_status st = ggml_backend_tensor_alloc(buf, m2_tensor, addr);
                    m2_bound = (st == GGML_STATUS_SUCCESS);
                    if (!m2_bound)
                        invalidate_cached_buffer(m2_quant.data);
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

        if (!m2_bound || m2_needs_upload)
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
        bool src_needs_upload = false;
        {
            ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
            if (dev != nullptr && src_quant.raw_bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                if (try_get_cacheable_tensor_buffer(
                        g_backend,
                        dev,
                        src_tensor,
                        src_quant.data,
                        static_cast<std::size_t>(src_quant.raw_bytes),
                        buf,
                        addr,
                        src_needs_upload))
                {
                    ggml_status st = ggml_backend_tensor_alloc(buf, src_tensor, addr);
                    src_bound = (st == GGML_STATUS_SUCCESS);
                    if (!src_bound)
                        invalidate_cached_buffer(src_quant.data);
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
        if (!src_bound || src_needs_upload)
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

} // anonymous namespace

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
