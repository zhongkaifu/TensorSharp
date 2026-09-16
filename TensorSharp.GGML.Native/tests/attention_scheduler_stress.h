// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Explicit two-device stress: graph allocation reuse, permuted Q, tiled
// attention and a following peer copy. The oracle is an exact masked mean.
static int scheduler_stress(int requested_queries, int requested_keys, int heads, int repeats,
    bool complete_placement, bool change_shapes, int sparse_capacity = 0)
{
    require(requested_queries > 0 && requested_keys >= 2 && heads > 0 && repeats > 0, "Invalid scheduler stress dimensions");
    if (ggml_backend_cuda_get_device_count() < 2) return 77;
    ggml_backend_t cuda[2] = {ggml_backend_cuda_init(0), ggml_backend_cuda_init(1)};
    require(cuda[0] && cuda[1], "Cannot initialize scheduler stress devices");
    ggml_backend_t backends[3] = {tsg_dsv4_fused_backend_init(cuda[0]),
        tsg_dsv4_fused_backend_init(cuda[1]), ggml_backend_cpu_init()};
    ggml_backend_buffer_type_t types[3];
    for (int i = 0; i < 3; ++i) {
        require(backends[i] != nullptr, "Cannot initialize scheduler stress backend");
        types[i] = ggml_backend_get_default_buffer_type(backends[i]);
    }
    constexpr int width = 512;
    std::vector<std::pair<int, int>> shapes = {{requested_queries, requested_keys}};
    if (change_shapes) {
        shapes = {{std::min(requested_queries, 17), std::min(requested_keys, 768)},
            {requested_queries, requested_keys},
            {std::max(1, requested_queries - 1), std::max(2, requested_keys - 1)},
            {4, requested_keys}, {2, std::min(requested_keys, 768)},
            {requested_queries, requested_keys}};
    }
    for (const auto & shape : shapes) {
    const int queries = shape.first, keys = shape.second;
    for (int target : {0, 1}) {
        auto * inputs = ggml_init({1024 * 1024, nullptr, true});
        auto * source = ggml_init({1024 * 1024, nullptr, true});
        auto * ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
        require(inputs && source && ctx, "Cannot initialize scheduler stress metadata");
        auto * q0 = ggml_new_tensor_3d(inputs, GGML_TYPE_F32, width, heads, queries);
        auto * mask = ggml_new_tensor_2d(inputs, GGML_TYPE_F16, keys, queries);
        auto * sinks = ggml_new_tensor_1d(inputs, GGML_TYPE_F32, heads);
        auto * kv = ggml_new_tensor_3d(source, GGML_TYPE_F16, width, keys, 1);
        for (auto * t : {q0, mask, sinks, kv}) ggml_set_input(t);
        auto * input_buffer = ggml_backend_alloc_ctx_tensors(inputs, cuda[target]);
        auto * source_buffer = ggml_backend_alloc_ctx_tensors(source, cuda[1 - target]);
        require(input_buffer && source_buffer, "Cannot allocate scheduler stress inputs");
        auto * scheduler = ggml_backend_sched_new(backends, types, 3, 8192, false, true);
        require(scheduler != nullptr, "Cannot initialize attention scheduler");
        auto * q = ggml_permute(ctx, q0, 0, 2, 1, 3);
        auto * output = complete_placement
            ? tsg_attention_f32_on_backend(ctx, scheduler, backends[target], q, kv, kv, mask, sinks,
                1.0f / std::sqrt(float(width)), sparse_capacity)
            : tsg_attention_f32(ctx, q, kv, kv, mask, sinks, 1.0f / std::sqrt(float(width)));
        if (!complete_placement) {
            // Retain the original partial-pin integration as a diagnostic
            // control, including its scheduler-placement failure behavior.
            ggml_backend_sched_set_tensor_backend(scheduler, output->src[0], backends[target]);
            ggml_backend_sched_set_tensor_backend(scheduler, output, backends[target]);
        }
        auto * result = ggml_scale(ctx, output, 0.5f);
        ggml_backend_sched_set_tensor_backend(scheduler, result, backends[1 - target]);
        ggml_set_output(result);
        auto * graph = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(graph, result);
        require(ggml_backend_sched_alloc_graph(scheduler, graph), "Cannot allocate scheduled attention graph");
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_CUSTOM) {
                require(ggml_backend_sched_get_tensor_backend(scheduler, node) != backends[2],
                    "Scheduler stress cannot silently run custom attention on CPU");
                if (complete_placement) {
                    require(ggml_backend_sched_get_tensor_backend(scheduler, node) == backends[target],
                        "Owned attention compute moved off its requested device");
                    require(node->buffer && ggml_backend_buffer_get_type(node->buffer) == types[target],
                        "Owned attention output storage differs from its compute device");
                }
            }
        }
        if (complete_placement)
            require(ggml_backend_buffer_get_type(output->buffer) == types[target],
                "Attention in-place output storage moved off its requested device");
        std::vector<float> zeros(size_t(width) * heads * queries, 0.0f);
        ggml_backend_tensor_set(q0, zeros.data(), 0, zeros.size() * sizeof(float));
        ggml_backend_tensor_set(sinks, zeros.data(), 0, heads * sizeof(float));
        std::vector<ggml_fp16_t> key_values(size_t(width) * keys);
        std::vector<ggml_fp16_t> mask_values(size_t(keys) * queries);
        std::vector<float> actual(size_t(width) * heads * queries);
        auto sample = [](int key, int d, int repeat) {
            return float((key * 13 + d * 7 + repeat * 3) % 31 - 15) / 32.0f;
        };
        for (int repeat = 0; repeat < repeats; ++repeat) {
            for (int key = 0; key < keys; ++key) for (int d = 0; d < width; ++d)
                key_values[size_t(key) * width + d] = ggml_fp32_to_fp16(sample(key, d, repeat));
            std::fill(mask_values.begin(), mask_values.end(), ggml_fp32_to_fp16(-INFINITY));
            for (int query = 1; query < queries; ++query) {
                const int first = (query + repeat) % keys, second = (first + 1) % keys;
                mask_values[size_t(query) * keys + first] = ggml_fp32_to_fp16(0);
                mask_values[size_t(query) * keys + second] = ggml_fp32_to_fp16(0);
            }
            ggml_backend_tensor_set(kv, key_values.data(), 0, key_values.size() * sizeof(ggml_fp16_t));
            ggml_backend_tensor_set(mask, mask_values.data(), 0, mask_values.size() * sizeof(ggml_fp16_t));
            require(ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS,
                "Scheduled attention compute failed");
            ggml_backend_sched_synchronize(scheduler);
            ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));
            double max_error = 0;
            for (int query = 0; query < queries; ++query) for (int head = 0; head < heads; ++head)
            for (int d = 0; d < width; ++d) {
                const int first = (query + repeat) % keys, second = (first + 1) % keys;
                const double expected = query ? (sample(first, d, repeat) + sample(second, d, repeat)) / 6.0 : 0.0;
                const float value = actual[(size_t(query) * heads + head) * width + d];
                require(std::isfinite(value), "Nonfinite scheduled attention result");
                max_error = std::max(max_error, std::abs(double(value) - expected));
            }
            std::printf("ATTENTION_SCHEDULER placement=%s target=%d queries=%d keys=%d heads=%d repeat=%d max_abs=%.8g splits=%d sparse_capacity=%d\n",
                complete_placement ? "complete" : "legacy-partial", target, queries, keys, heads, repeat,
                max_error, ggml_backend_sched_get_n_splits(scheduler), sparse_capacity);
            require(max_error <= 3e-6, "Scheduled attention changed masked means or output positions");
        }
        ggml_backend_sched_free(scheduler);
        ggml_backend_buffer_free(source_buffer);
        ggml_backend_buffer_free(input_buffer);
        ggml_free(ctx); ggml_free(source); ggml_free(inputs);
    }
    }
    for (auto * backend : backends) ggml_backend_free(backend);
    for (auto * backend : cuda) ggml_backend_free(backend);
    return 0;
}
