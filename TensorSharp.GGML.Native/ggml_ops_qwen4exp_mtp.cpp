// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Qwen4Exp NextN graph semantics were checked against the MIT-licensed
// danielhanchen/llama.cpp qwen4exp/mtp branch, commit
// d1a92352cbd417fd840b4e765c0b82f5fe3d1d89. This executor uses TensorSharp's
// existing HC/attention/MoE builders and owns its graph/allocation lifetime.
#include "ggml_ops_internal.h"
#include "ggml_ops_matmul_precision.h"
#include "ggml_ops_dsv4_fused.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <cmath>
#include <limits>

namespace
{
    void validate_matrix(int type, long long bytes, int64_t columns, int64_t rows, int64_t experts = 1)
    {
        if (type < 0 || type >= GGML_TYPE_COUNT || bytes <= 0 || columns <= 0 || rows <= 0 || experts <= 0)
            throw std::invalid_argument("qwen4exp MTP: invalid matrix storage");
        const auto* traits = ggml_get_type_traits((ggml_type)type);
        const bool floating = type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
        if ((!floating && !(traits->is_quantized && traits->to_float))
            || traits->blck_size <= 0 || traits->type_size == 0 || columns % traits->blck_size != 0)
            throw std::invalid_argument("qwen4exp MTP: unsupported matrix type or block alignment");
        const auto limit = (uint64_t)std::numeric_limits<long long>::max();
        uint64_t expected = (uint64_t)columns / traits->blck_size;
        for (uint64_t factor : {(uint64_t)traits->type_size, (uint64_t)rows, (uint64_t)experts})
        {
            if (expected > limit / factor)
                throw std::invalid_argument("qwen4exp MTP: matrix storage overflow");
            expected *= factor;
        }
        if ((uint64_t)bytes != expected)
            throw std::invalid_argument("qwen4exp MTP: matrix byte count does not match dimensions");
    }

    void validate_storage(const TSGgmlQwen4ExpMtpConfig& c, const TSGgmlQwen4ExpAttnArgs& a,
        const TSGgmlQwen4ExpFfnArgs& f, const TSGgmlQwen4ExpHeadArgs& h)
    {
        const int64_t width = (int64_t)c.n_embd * c.hc;
        const int64_t q_width = (int64_t)c.head_dim * c.n_head;
        const int64_t kv_width = (int64_t)c.head_dim * c.n_head_kv;
        if (c.n_embd > std::numeric_limits<int>::max() / 2
            || q_width > std::numeric_limits<int>::max() / 2
            || kv_width > std::numeric_limits<int>::max())
            throw std::invalid_argument("qwen4exp MTP: projection dimension overflow");
        validate_matrix(c.eh_type, c.eh_bytes, 2LL * c.n_embd, c.n_embd);
        validate_matrix(a.hc_down_type, a.hc_down_bytes, width, c.hc_low_rank);
        validate_matrix(a.hc_up_type, a.hc_up_bytes, c.hc_low_rank, width);
        validate_matrix(a.hc_inject_type, a.hc_inject_bytes, width, c.hc);
        validate_matrix(a.wq_type, a.wq_bytes, c.n_embd, q_width * 2);
        validate_matrix(a.wk_type, a.wk_bytes, c.n_embd, kv_width);
        validate_matrix(a.wv_type, a.wv_bytes, c.n_embd, kv_width);
        validate_matrix(a.wo_type, a.wo_bytes, q_width, c.n_embd);
        if (a.kv_type != GGML_TYPE_F32 && a.kv_type != GGML_TYPE_F16)
            throw std::invalid_argument("qwen4exp MTP: KV storage must be F32 or F16");
        validate_matrix(a.kv_type, a.kv_bytes, c.head_dim, c.capacity, c.n_head_kv);
        validate_matrix(f.hc_down_type, f.hc_down_bytes, width, c.hc_low_rank);
        validate_matrix(f.hc_up_type, f.hc_up_bytes, c.hc_low_rank, width);
        validate_matrix(f.hc_inject_type, f.hc_inject_bytes, width, c.hc);
        validate_matrix(f.router_type, f.router_bytes, c.n_embd, c.n_expert);
        validate_matrix(f.gate_exps_type, f.gate_exps_bytes, c.n_embd, c.n_ff, c.n_expert);
        validate_matrix(f.up_exps_type, f.up_exps_bytes, c.n_embd, c.n_ff, c.n_expert);
        validate_matrix(f.down_exps_type, f.down_exps_bytes, c.n_ff, c.n_embd, c.n_expert);
        validate_matrix(f.sh_gate_type, f.sh_gate_bytes, c.n_embd, c.n_ff_sh);
        validate_matrix(f.sh_up_type, f.sh_up_bytes, c.n_embd, c.n_ff_sh);
        validate_matrix(f.sh_down_type, f.sh_down_bytes, c.n_ff_sh, c.n_embd);
        validate_matrix(h.hc_down_type, h.hc_down_bytes, width, c.hc_low_rank);
        validate_matrix(h.hc_up_type, h.hc_up_bytes, c.hc_low_rank, width);
        validate_matrix(h.head_type, h.head_bytes, c.n_embd, h.vocab);
        int64_t sections = 0;
        for (int section : c.rope_sections)
        {
            if (section < 0) throw std::invalid_argument("qwen4exp MTP: negative rotary section");
            sections += section;
        }
        if (sections > c.n_rot)
            throw std::invalid_argument("qwen4exp MTP: rotary sections exceed rotary width");
    }

    struct Q4eMtpExecutor
    {
        TSGgmlQwen4ExpMtpConfig config{};
        TSGgmlQwen4ExpAttnArgs attn{};
        TSGgmlQwen4ExpFfnArgs ffn{};
        TSGgmlQwen4ExpHeadArgs head{};
        ggml_context* ctx = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_gallocr_t alloc = nullptr;
        // The shared device backend owns allocation and outlives this executor.
        // Its wrapper supplies the owned F32 operations without changing the
        // math mode of other model executions on the same backend.
        ggml_backend_t precise_backend = nullptr;
        ggml_context* kv_ctx = nullptr;
        ggml_backend_buffer_t kv_buffer = nullptr;
        ggml_tensor* owned_k = nullptr;
        ggml_tensor* owned_v = nullptr;
        ggml_tensor* embedding = nullptr;
        ggml_tensor* previous = nullptr;
        ggml_tensor* hidden = nullptr;
        ggml_tensor* logits = nullptr;
        ggml_tensor* mask = nullptr;
        ggml_tensor* positions = nullptr;
        ggml_tensor* rows = nullptr;
        std::vector<Q4eCachedBind> bindings;
        std::vector<ggml_tensor*> kv;
        int tokens = 0;
        int keys = 0;
        bool mrope = false;
        bool with_head = false;
        bool failed = false;

        void reset_graph()
        {
            if (alloc) { ggml_gallocr_free(alloc); alloc = nullptr; }
            if (ctx) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr;
            embedding = previous = hidden = logits = mask = positions = rows = nullptr;
            bindings.clear();
            kv.clear();
            tokens = keys = 0;
        }
        ~Q4eMtpExecutor()
        {
            if (precise_backend)
            {
                ggml_backend_synchronize(precise_backend);
                ggml_backend_free(precise_backend);
            }
            reset_graph();
            if (kv_buffer) ggml_backend_buffer_free(kv_buffer);
            if (kv_ctx) ggml_free(kv_ctx);
        }

        ggml_backend_t execution_backend() const
        {
            return precise_backend ? precise_backend : tsg::active_backend();
        }

        void initialize_kv()
        {
            ggml_init_params ip{};
            ip.mem_size = 2 * ggml_tensor_overhead();
            ip.no_alloc = true;
            kv_ctx = ggml_init(ip);
            if (!kv_ctx) throw std::bad_alloc();
            owned_k = ggml_new_tensor_3d(kv_ctx, (ggml_type)attn.kv_type,
                    config.head_dim, config.capacity, config.n_head_kv);
            owned_v = ggml_new_tensor_3d(kv_ctx, (ggml_type)attn.kv_type,
                    config.head_dim, config.capacity, config.n_head_kv);
            kv_buffer = ggml_backend_alloc_ctx_tensors(kv_ctx, tsg::active_backend());
            if (!kv_buffer) throw std::bad_alloc();
            // The caller seeds a new/grown cache once. These private buffers
            // outlive every graph and never depend on host registration or a
            // weight-cache admission policy to retain mutable state.
            ggml_backend_tensor_set(owned_k, attn.k_cache, 0, (std::size_t)attn.kv_bytes);
            ggml_backend_tensor_set(owned_v, attn.v_cache, 0, (std::size_t)attn.kv_bytes);
        }

        bool bindings_current()
        {
            for (const auto& binding : bindings)
            {
                void* old = binding.tensor->data;
                bool upload = false;
                if (!tsg::try_bind_cached_tensor(tsg::active_backend(),
                        ggml_backend_get_device(tsg::active_backend()), binding.tensor,
                        binding.data, binding.bytes, upload, binding.usage)
                    || old != binding.tensor->data)
                    return false;
                if (upload)
                    ggml_backend_tensor_set(binding.tensor, tsg::resolve_upload_source(binding.data), 0, binding.bytes);
            }
            return true;
        }

        void build(int count, int padded_keys, bool use_mrope, bool output_head)
        {
            if (graph) tsg::sync_backend(tsg::active_backend());
            reset_graph();
            const auto& c = config;
            const int width = c.n_embd * c.hc;
            ggml_init_params ip{};
            ip.mem_size = ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(2048, false);
            ip.no_alloc = true;
            ctx = ggml_init(ip);
            if (!ctx) throw std::bad_alloc();
            graph = ggml_new_graph_custom(ctx, 2048, false);
            Q4eBinder binder{ggml_backend_get_device(tsg::active_backend())};
            embedding = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.n_embd, count);
            previous = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, count);
            mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, padded_keys, count);
            positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (use_mrope ? 4 : 1) * count);
            rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, count);
            for (auto* input : {embedding, previous, mask, positions, rows}) ggml_set_input(input);

            auto* enorm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.n_embd);
            auto* hnorm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
            auto* eh = ggml_new_tensor_2d(ctx, (ggml_type)c.eh_type, 2 * c.n_embd, c.n_embd);
            binder.add(enorm, c.enorm, (std::size_t)c.n_embd * sizeof(float));
            binder.add(hnorm, c.hnorm, (std::size_t)width * sizeof(float));
            binder.add(eh, c.eh_proj, (std::size_t)c.eh_bytes);
            auto* e = ggml_mul(ctx, ggml_rms_norm(ctx, embedding, c.eps), enorm);
            auto* h3 = ggml_reshape_3d(ctx, previous, c.n_embd, c.hc, count);
            auto* h = ggml_reshape_3d(ctx, ggml_mul(ctx,
                    ggml_reshape_2d(ctx, ggml_rms_norm(ctx, h3, c.eps), width, count), hnorm),
                    c.n_embd, c.hc, count);
            e = ggml_repeat(ctx, ggml_reshape_3d(ctx, e, c.n_embd, 1, count), h);
            // Each stream sees [normalized token embedding, normalized stream].
            // A single RMS over width, or concatenating all embeddings first,
            // implements a different trained model.
            auto* joined = ggml_reshape_2d(ctx, ggml_concat(ctx, e, h, 0), 2 * c.n_embd, c.hc * count);
            auto* residual = ggml_reshape_2d(ctx, ggml_mul_mat(ctx, eh, joined), width, count);
            residual = q4e_nodes_attn(ctx, graph, binder, &attn, residual, mask, positions, rows,
                    c.n_embd, c.hc, c.hc_low_rank, count, c.head_dim, c.n_head, c.n_head_kv,
                    c.capacity, padded_keys, c.n_rot, c.rope_base, c.rope_scale, c.attn_scale,
                    c.eps, q4e_flash_attn_ok(attn.kv_type, c.head_dim), &kv, nullptr,
                    use_mrope ? c.rope_sections : nullptr, nullptr, owned_k, owned_v);
            residual = q4e_nodes_ffn(ctx, binder, &ffn, residual,
                    c.n_embd, c.hc, c.hc_low_rank, count,
                    c.n_expert, c.n_expert_used, c.n_ff, c.n_ff_sh, c.eps);
            // Keep the actual producer alive for chained drafting.
            hidden = residual;
            ggml_set_output(hidden);
            ggml_build_forward_expand(graph, hidden);
            if (output_head)
            {
                auto* last = ggml_cont(ctx, ggml_view_2d(ctx, hidden, width, 1,
                        hidden->nb[1], (std::size_t)(count - 1) * hidden->nb[1]));
                logits = q4e_nodes_head(ctx, binder, &head, last,
                        c.n_embd, c.hc, c.hc_low_rank, 1, c.eps);
                ggml_set_output(logits);
                ggml_build_forward_expand(graph, logits);
            }
            if (precise_backend)
            {
                // F32 graph operands must not silently become TF32 in cuBLAS
                // or inline MMA dispatch. Keep quantized/mixed-weight arithmetic
                // on its declared route; the exact floating helper owns its
                // cuBLAS handle and handles ordinary and indexed matmuls.
                for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
                {
                    auto* node = ggml_graph_node(graph, i);
                    if ((node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID)
                        && node->src[0]->type == GGML_TYPE_F32
                        && node->src[1]->type == GGML_TYPE_F32)
                        tsg_matmul_require_f32(ctx, node);
                }
            }
            alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(tsg::active_backend()));
            if (!alloc || !ggml_gallocr_alloc_graph(alloc, graph))
                throw std::runtime_error("qwen4exp MTP graph allocation failed");
            binder.flush();
            bindings = std::move(binder.cached);
            tokens = count;
            keys = padded_keys;
            mrope = use_mrope;
            with_head = output_head;
        }
    };
}

TSG_EXPORT void* TSGgml_Qwen4ExpMtpCreate(const TSGgmlQwen4ExpMtpConfig* config,
    const TSGgmlQwen4ExpAttnArgs* attn, const TSGgmlQwen4ExpFfnArgs* ffn,
    const TSGgmlQwen4ExpHeadArgs* head)
{
    try
    {
        tsg::set_last_error("");
        if (!config || !attn || !ffn || !head || !config->enorm || !config->hnorm || !config->eh_proj
            || config->n_embd <= 0 || config->hc <= 0 || config->hc_low_rank <= 0
            || config->n_embd > std::numeric_limits<int>::max() / config->hc
            || config->capacity <= 0 || config->head_dim <= 0 || config->n_head_kv <= 0
            || config->hc < 2 || config->n_rot <= 0 || config->n_rot > config->head_dim
            || config->n_rot % 2 != 0 || config->n_ff <= 0 || config->n_ff_sh <= 0
            || !std::isfinite(config->eps) || config->eps <= 0
            || !std::isfinite(config->rope_base) || config->rope_base <= 0
            || !std::isfinite(config->rope_scale) || config->rope_scale <= 0
            || !std::isfinite(config->attn_scale) || config->attn_scale <= 0
            || config->n_head <= 0 || config->n_head % config->n_head_kv != 0
            || config->n_expert_used <= 0 || config->n_expert_used > config->n_expert
            || head->vocab <= 0 || !head->head || !attn->k_cache || !attn->v_cache
            || config->device < 0 || config->device >= tsg::g_device_count.load(std::memory_order_acquire))
            throw std::invalid_argument("qwen4exp MTP: invalid descriptors or dimensions");
        for (const void* pointer : {attn->hc_norm, attn->hc_down, attn->hc_up, attn->hc_inject,
                attn->wq, attn->wk, attn->wv, attn->wo, attn->q_norm, attn->k_norm,
                ffn->hc_norm, ffn->hc_down, ffn->hc_up, ffn->hc_inject, ffn->router,
                ffn->gate_exps, ffn->up_exps, ffn->down_exps, ffn->sh_gate_inp,
                ffn->sh_gate, ffn->sh_up, ffn->sh_down, head->hc_norm, head->hc_down, head->hc_up})
            if (pointer == nullptr) throw std::invalid_argument("qwen4exp MTP: null weight binding");
        validate_storage(*config, *attn, *ffn, *head);
        auto executor = std::make_unique<Q4eMtpExecutor>();
        executor->config = *config;
        executor->attn = *attn;
        executor->ffn = *ffn;
        executor->head = *head;
        tsg::ScopedRank rank(config->device);
        if (!tsg::ensure_backend()) return nullptr;
#ifdef TSG_GGML_USE_CUDA
        if (ggml_backend_is_cuda(tsg::active_backend()))
        {
            executor->precise_backend = tsg_dsv4_fused_backend_init(tsg::active_backend());
            if (!executor->precise_backend)
                throw std::runtime_error("qwen4exp MTP: cannot create F32 execution backend");
        }
#endif
        executor->initialize_kv();
        return executor.release();
    }
    catch (const std::exception& e) { tsg::set_last_error(e.what()); return nullptr; }
    catch (...) { tsg::set_last_error("qwen4exp MTP: creation failed"); return nullptr; }
}

TSG_EXPORT int TSGgml_Qwen4ExpMtpForward(void* handle,
    const float* embeddings, const float* previous, int count, int position,
    int rope_position, const int* mrope3, float* hidden_out, float* logits_out)
{
    auto* executor = static_cast<Q4eMtpExecutor*>(handle);
    try
    {
        tsg::set_last_error("");
        if (!executor || executor->failed || !embeddings || !previous || count <= 0 || position < 0
            || position > executor->config.capacity || count > executor->config.capacity - position
            || rope_position < 0 || count > std::numeric_limits<int>::max() - rope_position
            || count > std::numeric_limits<int>::max() / 4
            || count > std::numeric_limits<int>::max() / executor->config.hc)
            throw std::invalid_argument("qwen4exp MTP: invalid forward or failed executor");
        const auto& c = executor->config;
        if (mrope3 != nullptr && c.rope_sections[0] == 0 && c.rope_sections[1] == 0 && c.rope_sections[2] == 0)
            throw std::invalid_argument("qwen4exp MTP: multi-axis positions require active rotary sections");
        tsg::ScopedRank rank(c.device);
        if (!tsg::ensure_backend()) return 0;
        const bool fa = q4e_flash_attn_ok(executor->attn.kv_type, c.head_dim);
        const int total = position + count;
        const int padded = fa ? (int)std::min<int64_t>(c.capacity, ((int64_t)total + 255) / 256 * 256) : total;
        if (!executor->graph || executor->tokens != count || executor->keys != padded
            || executor->mrope != (mrope3 != nullptr) || executor->with_head != (logits_out != nullptr)
            || !executor->bindings_current())
            executor->build(count, padded, mrope3 != nullptr, logits_out != nullptr);
        if (position == 0)
            for (auto* tensor : executor->kv) ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
        std::vector<ggml_fp16_t> mask((std::size_t)padded * count);
        std::vector<int64_t> rows(count);
        std::vector<int32_t> positions((mrope3 ? 4 : 1) * (std::size_t)count, 0);
        for (int t = 0; t < count; ++t)
        {
            rows[t] = position + t;
            for (int k = 0; k < padded; ++k)
                mask[(std::size_t)t * padded + k] = k <= position + t ? 0 : (ggml_fp16_t)0xfc00;
            if (mrope3)
                for (int axis = 0; axis < 3; ++axis) positions[axis * count + t] = mrope3[3 * t + axis];
            else positions[t] = rope_position + t;
        }
        ggml_backend_tensor_set(executor->embedding, embeddings, 0, (std::size_t)c.n_embd * count * sizeof(float));
        ggml_backend_tensor_set(executor->previous, previous, 0, (std::size_t)c.n_embd * c.hc * count * sizeof(float));
        ggml_backend_tensor_set(executor->mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(executor->rows, rows.data(), 0, rows.size() * sizeof(int64_t));
        ggml_backend_tensor_set(executor->positions, positions.data(), 0, positions.size() * sizeof(int32_t));
        if (tsg::graph_compute_profiled(executor->execution_backend(), executor->graph, "qwen4exp MTP") != GGML_STATUS_SUCCESS)
            throw std::runtime_error("qwen4exp MTP: graph execution failed");
        if (hidden_out)
            ggml_backend_tensor_get(executor->hidden, hidden_out,
                    (std::size_t)(count - 1) * c.n_embd * c.hc * sizeof(float),
                    (std::size_t)c.n_embd * c.hc * sizeof(float));
        if (logits_out)
            ggml_backend_tensor_get(executor->logits, logits_out, 0, (std::size_t)executor->head.vocab * sizeof(float));
        return 1;
    }
    catch (const std::exception& e)
    {
        if (executor) executor->failed = true;
        tsg::set_last_error(e.what());
        return 0;
    }
    catch (...)
    {
        if (executor) executor->failed = true;
        tsg::set_last_error("qwen4exp MTP: forward failed");
        return 0;
    }
}

TSG_EXPORT int TSGgml_Qwen4ExpMtpCopyKv(void* handle, void* k, void* v, long long bytes)
{
    auto* executor = static_cast<Q4eMtpExecutor*>(handle);
    try
    {
        tsg::set_last_error("");
        if (!executor || executor->failed || !k || !v || bytes != executor->attn.kv_bytes)
            throw std::invalid_argument("qwen4exp MTP: invalid KV export");
        tsg::ScopedRank rank(executor->config.device);
        ggml_backend_tensor_get(executor->owned_k, k, 0, (std::size_t)bytes);
        ggml_backend_tensor_get(executor->owned_v, v, 0, (std::size_t)bytes);
        return 1;
    }
    catch (const std::exception& e) { tsg::set_last_error(e.what()); return 0; }
    catch (...) { tsg::set_last_error("qwen4exp MTP: KV export failed"); return 0; }
}

TSG_EXPORT void TSGgml_Qwen4ExpMtpFree(void* handle)
{
    try
    {
        auto* executor = static_cast<Q4eMtpExecutor*>(handle);
        if (executor != nullptr)
        {
            tsg::ScopedRank rank(executor->config.device);
            tsg::sync_backend(tsg::active_backend());
            delete executor;
        }
    }
    catch (...) { tsg::set_last_error("qwen4exp MTP: release failed"); }
}
