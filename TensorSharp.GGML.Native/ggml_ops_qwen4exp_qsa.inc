// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// QSA semantics checked against MIT llama.cpp d1d3c3396aa13a5f239109a822666c4870490ad5.
// Uses public ggml nodes, with no graph callbacks or external source changes.
static ggml_tensor* q4e_qsa_matmul(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x)
{
    auto* out = ggml_mul_mat(ctx, w, x);
#ifdef TSG_GGML_USE_CUDA
    if (ggml_backend_is_cuda(g_backend) && w->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32)
        tsg_matmul_require_f32(ctx, out);
#endif
    return out;
}

static ggml_tensor* q4e_nodes_qsa_mask(ggml_context* ctx, ggml_cgraph* graph,
    Q4eBinder& binder, const Q4eQsaGraph& qsa, ggml_tensor* mixed,
    ggml_tensor* original_mask, ggml_tensor* kv_idx,
    int n_embd, int T, int padded, int n_rot, float base, float scale, float eps)
{
    const auto& a = *qsa.args;
    const auto& in = *qsa.inputs;
    const int D = a.head_dim, H = a.heads, R = a.ratio;
    const int B = (padded + R - 1) / R;
    auto* wk = ggml_new_tensor_2d(ctx, (ggml_type)a.k_type, n_embd, D);
    auto* wq = ggml_new_tensor_2d(ctx, (ggml_type)a.q_type, n_embd, (int64_t)D * H);
    auto* nk = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    auto* nq = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    binder.add(wk, a.k_proj, (size_t)a.k_bytes);
    auto* raw = q4e_qsa_matmul(ctx, wk, mixed);
    // A real dependency from the set_rows result sequences append before read.
    auto* written = ggml_set_rows(ctx, qsa.cache, raw, kv_idx);
    ggml_build_forward_expand(graph, written);
    if (padded <= (int64_t)a.top_k + R - 1) return original_mask;
    binder.add(wq, a.q_proj, (size_t)a.q_bytes);
    binder.add(nk, a.k_norm, (size_t)D * sizeof(float));
    binder.add(nq, a.q_norm, (size_t)D * sizeof(float));
    auto* keys = ggml_view_2d(ctx, written, D, padded, written->nb[1], 0);
    auto* members = ggml_reshape_3d(ctx, ggml_get_rows(ctx, keys, in.block_cells), D, R, B);
    ggml_tensor* pooled = nullptr;
    for (int r = 0; r < R; ++r)
    {
        auto* part = ggml_cont(ctx, ggml_view_2d(ctx, members, D, B, members->nb[2], r * members->nb[1]));
        pooled = pooled ? ggml_add(ctx, pooled, part) : part;
    }
    pooled = ggml_scale(ctx, pooled, 1.f / R);
    if (qsa.probes) { ggml_set_output(pooled); qsa.probes->push_back(pooled); }
    pooled = ggml_mul(ctx, ggml_rms_norm(ctx, pooled, eps), nk);
    pooled = ggml_reshape_3d(ctx, pooled, D, 1, B);
    int sections[4] = {a.rope_sections[0], a.rope_sections[1], a.rope_sections[2], a.rope_sections[3]};
    pooled = ggml_rope_multi(ctx, pooled, in.block_positions, nullptr, n_rot, sections,
        GGML_ROPE_TYPE_IMROPE, 0, base, scale, 0.f, 1.f, 0.f, 0.f);
    pooled = ggml_reshape_2d(ctx, pooled, D, B);
    if (qsa.probes) { ggml_set_output(pooled); qsa.probes->push_back(pooled); }
    auto* q = ggml_reshape_3d(ctx, q4e_qsa_matmul(ctx, wq, mixed), D, H, T);
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, eps), nq);
    q = ggml_rope_multi(ctx, q, in.query_positions, nullptr, n_rot, sections,
        GGML_ROPE_TYPE_IMROPE, 0, base, scale, 0.f, 1.f, 0.f, 0.f);
    if (qsa.probes) { ggml_set_output(q); qsa.probes->push_back(q); }
    auto* scores = ggml_reshape_3d(ctx,
        q4e_qsa_matmul(ctx, pooled, ggml_reshape_2d(ctx, q, D, (int64_t)H * T)), B, H, T);
    scores = ggml_relu(ctx, scores);
    ggml_tensor* sum = nullptr;
    for (int h = 0; h < H; ++h)
    {
        auto* part = ggml_cont(ctx, ggml_view_2d(ctx, scores, B, T, scores->nb[2], h * scores->nb[1]));
        sum = sum ? ggml_add(ctx, sum, part) : part;
    }
    if (qsa.probes) { ggml_set_output(sum); qsa.probes->push_back(sum); }
    sum = ggml_add(ctx, sum, in.bias);
    auto* expanded = ggml_get_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, sum)), in.cell_blocks);
    expanded = ggml_add(ctx, ggml_cont(ctx, ggml_transpose(ctx, expanded)), ggml_cast(ctx, original_mask, GGML_TYPE_F32));
    const int width = (int)std::min<int64_t>(padded, (int64_t)a.top_k + R - 1);
    auto* selected = ggml_cont(ctx, ggml_top_k(ctx, expanded, width));
    auto* all = ggml_fill(ctx, original_mask, -INFINITY);
    auto* rows = ggml_view_3d(ctx, all, 1, padded, T, all->nb[0], all->nb[1], 0);
    auto* zeros = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, width, T), 0.f);
    auto* sparse = ggml_set_rows(ctx, rows, zeros, selected);
    sparse = ggml_view_2d(ctx, sparse, padded, T, sparse->nb[2], 0);
    return ggml_add(ctx, sparse, original_mask);
}

#ifdef TSG_GGML_TEST_HOOKS
// Isolated arithmetic probe using the production builder. Buffers are supplied
// by the fixture; no target model or per-holder state is created here.
TSG_EXPORT int TSGgml_Qwen4ExpTestQsa(
    const TSGgmlQwen4ExpQsaArgs* args, const float* mixed_data, const uint16_t* mask_data,
    const int32_t* positions, int n_embd, int T, int capacity, int live, int padded, int start,
    int n_rot, float base, float scale, float eps, int device,
    uint16_t* mask_out, float* pooled_out, float* key_out, float* query_out, float* score_out)
{
    try
    {
        if (!args || !mixed_data || !mask_data || !mask_out || !pooled_out || !key_out || !query_out || !score_out
            || n_embd <= 0 || n_embd > 16384 || T <= 0 || T > 4096 || capacity < padded || padded > 65536
            || args->head_dim < n_rot || args->head_dim > 1024 || args->heads <= 0 || args->heads > 64
            || args->ratio <= 0 || args->ratio > 64 || n_rot <= 0 || n_rot % 2
            || !std::isfinite(base) || base <= 0 || !std::isfinite(scale) || scale <= 0 || !std::isfinite(eps) || eps <= 0
            || args->top_k <= 0 || args->top_k >= padded - args->ratio + 1
            || args->k_type != GGML_TYPE_F32 || args->q_type != GGML_TYPE_F32
            || (args->cache_type != GGML_TYPE_F16 && args->cache_type != GGML_TYPE_F32)
            || args->k_bytes != (int64_t)n_embd * args->head_dim * 4
            || args->q_bytes != (int64_t)n_embd * args->head_dim * args->heads * 4
            || args->cache_bytes != (int64_t)capacity * args->head_dim * (args->cache_type == GGML_TYPE_F16 ? 2 : 4)
            || !args->k_proj || !args->q_proj || !args->k_norm || !args->q_norm || !args->cache)
            throw std::invalid_argument("qwen4exp QSA test: invalid shape/storage");
        auto plan = q4e_qsa_plan(positions, live, padded, start, T, args->ratio);
        tsg::ScopedRank rank(q4e_resolve_device(device));
        if (!ensure_backend()) return 0;
        ggml_init_params params{ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(2048, false), nullptr, true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
        if (!ctx) throw std::bad_alloc();
        auto* graph = ggml_new_graph_custom(ctx.get(), 2048, false);
        auto* mixed = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, T);
        auto* mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, padded, T);
        auto* rows = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, T);
        auto* cache = ggml_new_tensor_2d(ctx.get(), (ggml_type)args->cache_type, args->head_dim, capacity);
        Q4eQsaInputs inputs;
        inputs.ratio = args->ratio;
        auto input = [&](ggml_type type, int64_t size) { auto* t = ggml_new_tensor_1d(ctx.get(), type, size); ggml_set_input(t); return t; };
        inputs.cell_blocks = input(GGML_TYPE_I32, padded);
        inputs.block_cells = input(GGML_TYPE_I32, plan.block_cells.size());
        inputs.block_positions = input(GGML_TYPE_I32, plan.block_positions.size());
        inputs.query_positions = input(GGML_TYPE_I32, plan.query_positions.size());
        const int blocks = (padded + args->ratio - 1) / args->ratio;
        inputs.bias = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, blocks, T);
        for (auto* t : {mixed, mask, rows, cache, inputs.bias}) ggml_set_input(t);
        Q4eBinder binder{ggml_backend_get_device(g_backend)};
        std::vector<ggml_tensor*> probes;
        Q4eQsaGraph qsa{args, &inputs, cache, &probes};
        auto* result = q4e_nodes_qsa_mask(ctx.get(), graph, binder, qsa, mixed, mask, rows,
            n_embd, T, padded, n_rot, base, scale, eps);
        ggml_set_output(result);
        ggml_build_forward_expand(graph, result);
        std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, decltype(&ggml_gallocr_free)>
            alloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend)), ggml_gallocr_free);
        if (!alloc || !ggml_gallocr_alloc_graph(alloc.get(), graph)) throw std::bad_alloc();
        std::unique_ptr<std::remove_pointer_t<ggml_backend_t>, decltype(&ggml_backend_free)>
            wrapper(nullptr, ggml_backend_free);
#ifdef TSG_GGML_USE_CUDA
        if (ggml_backend_is_cuda(g_backend))
        {
            wrapper.reset(tsg_dsv4_fused_backend_init(g_backend));
            if (!wrapper) throw std::bad_alloc();
        }
#endif
        binder.flush();
        ggml_backend_tensor_set(cache, args->cache, 0, (size_t)args->cache_bytes);
        ggml_backend_tensor_set(mixed, mixed_data, 0, ggml_nbytes(mixed));
        ggml_backend_tensor_set(mask, mask_data, 0, ggml_nbytes(mask));
        std::vector<int64_t> row_data(T);
        std::iota(row_data.begin(), row_data.end(), (int64_t)start);
        ggml_backend_tensor_set(rows, row_data.data(), 0, ggml_nbytes(rows));
        auto put = [](ggml_tensor* t, const auto& values) { ggml_backend_tensor_set(t, values.data(), 0, values.size() * sizeof(values[0])); };
        put(inputs.cell_blocks, plan.cell_blocks); put(inputs.block_cells, plan.block_cells);
        put(inputs.block_positions, plan.block_positions); put(inputs.query_positions, plan.query_positions); put(inputs.bias, plan.bias);
        if (ggml_backend_graph_compute(wrapper ? wrapper.get() : g_backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("qwen4exp QSA test: compute failed");
        tsg::sync_backend(g_backend);
        ggml_backend_tensor_get(result, mask_out, 0, ggml_nbytes(result));
        float* outputs[] = {pooled_out, key_out, query_out, score_out};
        if (probes.size() != 4) throw std::logic_error("qwen4exp QSA test: missing probes");
        for (size_t i = 0; i < probes.size(); ++i) ggml_backend_tensor_get(probes[i], outputs[i], 0, ggml_nbytes(probes[i]));
        return 1;
    }
    catch (const std::exception& e) { set_last_error(e.what()); return 0; }
    catch (...) { set_last_error("qwen4exp QSA test: unknown failure"); return 0; }
}
#endif
