// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_internal.h"
#include "ggml-backend-impl.h"
#include "gguf.h"
#include "ggml_ops_q8_precision.h"
#include "ggml_ops_dsv4_fused.h"

#include <cctype>
#include <fstream>
#include <map>

namespace tsg_embeddings {
namespace {
void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error("Embedding encoder: " + message);
}
struct source_file {
    gguf_context * file = nullptr;
    ggml_context * tensors = nullptr;
    ~source_file() { if (file) gguf_free(file); if (tensors) ggml_free(tensors); }
};
int integer(gguf_context * file, const std::string & name, int fallback = -1) {
    const auto key = gguf_find_key(file, name.c_str());
    if (key < 0) { require(fallback >= 0, "missing metadata " + name); return fallback; }
    int64_t value = -1;
    switch (gguf_get_kv_type(file, key)) {
        case GGUF_TYPE_UINT32: value = gguf_get_val_u32(file, key); break;
        case GGUF_TYPE_INT32: value = gguf_get_val_i32(file, key); break;
        case GGUF_TYPE_UINT64: value = int64_t(gguf_get_val_u64(file, key)); break;
        case GGUF_TYPE_INT64: value = gguf_get_val_i64(file, key); break;
        default: break;
    }
    require(value >= 0 && value <= INT32_MAX, "invalid metadata " + name);
    return int(value);
}
float number(gguf_context * file, const std::string & name, float fallback) {
    const auto key = gguf_find_key(file, name.c_str());
    if (key < 0) return fallback;
    require(gguf_get_kv_type(file, key) == GGUF_TYPE_FLOAT32, "invalid metadata " + name);
    return gguf_get_val_f32(file, key);
}
}

struct encoder {
    int dim = 0, heads = 0, layers = 0, ff = 0, context = 0, vocab = 0, pooling = 0;
    float eps = 0;
    ggml_backend_t backend = nullptr;
    // Optional owned CUDA execution wrapper; the raw backend remains the
    // allocator and capability identity, and must outlive this wrapper.
    ggml_backend_t precise_backend = nullptr;
    ggml_threadpool_t threadpool = nullptr;
    ggml_context * weights_ctx = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    std::vector<ggml_backend_buffer_t> optimized_buffers;
    int cpu_alignment = 16;
    std::map<std::string, ggml_tensor *> weights;
    std::mutex mutex;
    struct graph {
        ggml_context * ctx = nullptr;
        ggml_gallocr_t allocator = nullptr;
        ggml_cgraph * gf = nullptr;
        ggml_backend_t plan_backend = nullptr;
        ggml_backend_graph_plan_t cpu_plan = nullptr;
        ggml_tensor * tokens = nullptr, * positions = nullptr, * mask = nullptr;
        ggml_tensor * selected = nullptr, * mean_weights = nullptr, * output = nullptr;
        int length = 0, batch = 0, sequences = 0;
        bool packed = false;
        std::vector<int> segments;
        std::vector<ggml_tensor *> segment_masks;
        ~graph() {
            if (cpu_plan) ggml_backend_graph_plan_free(plan_backend, cpu_plan);
            if (allocator) ggml_gallocr_free(allocator);
            if (ctx) ggml_free(ctx);
        }
    };
    std::unique_ptr<graph> cached;
    ~encoder() {
        if (precise_backend) ggml_backend_synchronize(precise_backend);
        cached.reset();
        if (precise_backend) ggml_backend_free(precise_backend);
        if (weights_buffer) ggml_backend_buffer_free(weights_buffer);
        for (auto * buffer : optimized_buffers) ggml_backend_buffer_free(buffer);
        if (weights_ctx) ggml_free(weights_ctx);
        if (backend) ggml_backend_free(backend);
        if (threadpool) ggml_threadpool_free(threadpool);
    }
    ggml_tensor * find(const std::string & name) const {
        const auto found = weights.find(name);
        return found == weights.end() ? nullptr : found->second;
    }
    ggml_tensor * weight(const std::string & name, int64_t rows = -1, int64_t cols = -1) const {
        auto * value = find(name);
        require(value != nullptr, "missing tensor " + name);
        require((rows < 0 || value->ne[0] == rows) && (cols < 0 || value->ne[1] == cols) &&
                value->ne[2] == 1 && value->ne[3] == 1, "invalid dimensions for " + name);
        return value;
    }
    ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
        auto * w = weight(name + ".weight");
        auto * y = precise_backend && w->type == GGML_TYPE_Q8_0
            ? tsg_matmul_q8_f32(ctx, w, x) : ggml_mul_mat(ctx, w, x);
        if (auto * bias = find(name + ".bias")) y = ggml_add(ctx, y, bias);
        return y;
    }
    ggml_tensor * norm(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
        return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), weight(name + ".weight", dim, 1)),
                        weight(name + ".bias", dim, 1));
    }
    void validate() const {
        weight("token_embd.weight", dim, vocab);
        weight("position_embd.weight", dim, context);
        weight("token_embd_norm.weight", dim, 1); weight("token_embd_norm.bias", dim, 1);
        if (find("token_types.weight")) weight("token_types.weight", dim);
        auto projection = [&](const std::string & name, int input, int output) {
            weight(name + ".weight", input, output);
            if (find(name + ".bias")) weight(name + ".bias", output, 1);
        };
        for (int l = 0; l < layers; ++l) {
            const auto p = "blk." + std::to_string(l) + ".";
            if (find(p + "attn_qkv.weight")) projection(p + "attn_qkv", dim, 3 * dim);
            else for (const auto * part : {"attn_q", "attn_k", "attn_v"}) projection(p + part, dim, dim);
            projection(p + "attn_output", dim, dim);
            projection(p + "ffn_up", dim, ff); projection(p + "ffn_down", ff, dim);
            for (const auto * part : {"attn_output_norm", "layer_output_norm"}) {
                weight(p + part + ".weight", dim, 1); weight(p + part + ".bias", dim, 1);
            }
        }
    }
    ggml_tensor * attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                            ggml_tensor * mask, int length, int batch) const {
        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        k = ggml_permute(ctx, k, 0, 2, 1, 3);
        v = ggml_permute(ctx, v, 0, 2, 1, 3);
        // CPU's vector and tiled attention need F32 K/V for consistent accumulation.
        const auto kv_type = ggml_backend_is_cpu(backend) ? GGML_TYPE_F32 : GGML_TYPE_F16;
        // Short CPU attention accepts row-strided F32 views directly. For long
        // sequences, pack each head contiguously once instead of repeatedly
        // walking the fused QKV projection's 3*dim row stride for every tile.
        // Preserve F32 K/V so vector and tiled paths retain the same precision.
        if (ggml_backend_is_cpu(backend) && length >= 1024) {
            k = ggml_cont(ctx, k);
            v = ggml_cont(ctx, v);
        }
        auto * kh = k->type == kv_type ? k : ggml_cast(ctx, k, kv_type);
        auto * vh = v->type == kv_type ? v : ggml_cast(ctx, v, kv_type);
        const float scale = 1.0f / std::sqrt(float(dim / heads));
        auto * fa = ggml_flash_attn_ext(ctx, q, kh, vh, mask, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(fa, GGML_PREC_F32);
        if (ggml_backend_supports_op(backend, fa)) return ggml_reshape_2d(ctx, fa, dim, length * batch);
        auto * scores = ggml_mul_mat(ctx, k, q);
        auto * mask_f32 = mask ? ggml_cast(ctx, mask, GGML_TYPE_F32) : nullptr;
        scores = ggml_soft_max_ext(ctx, scores, mask_f32, scale, 0.0f);
        auto * out = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, v)), scores);
        return ggml_cont_2d(ctx, ggml_permute(ctx, out, 0, 2, 1, 3), dim, length * batch);
    }
    std::unique_ptr<graph> build(int length, int batch, int sequences, bool packed, const std::vector<int> & segments) const {
        auto g = std::make_unique<graph>();
        g->length = length; g->batch = batch; g->sequences = sequences; g->packed = packed;
        g->segments = segments;
        const int n = length * batch, head = dim / heads;
        const size_t nodes = size_t(layers) * (100 + 24 * segments.size()) + 256;
        g->ctx = ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false) + 4096, nullptr, true});
        require(g->ctx != nullptr, "cannot allocate graph metadata");
        auto * ctx = g->ctx;
        g->gf = ggml_new_graph_custom(ctx, nodes, false);
        g->tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        g->positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_set_input(g->tokens); ggml_set_input(g->positions);
        // Equal lengths use independent streams; packed inputs use a block mask.
        // Mask rows are padded for the flash-attention kernels on CUDA and Metal.
        const int mask_queries = ((length + 31) / 32) * 32;
        if (segments.empty() && (packed || ggml_backend_is_cpu(backend))) {
            g->mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, length, mask_queries, 1, batch);
            ggml_set_input(g->mask);
        }
        for (int size : segments) {
            auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, size, ((size + 31) / 32) * 32);
            ggml_set_input(mask); g->segment_masks.push_back(mask);
        }
        auto * x = ggml_get_rows(ctx, weight("token_embd.weight"), g->tokens);
        if (auto * types = find("token_types.weight")) {
            auto * first_type = ggml_view_1d(ctx, types, dim, 0);
            if (first_type->type != GGML_TYPE_F32) first_type = ggml_cast(ctx, first_type, GGML_TYPE_F32);
            x = ggml_add(ctx, x, first_type);
        }
        x = ggml_add(ctx, ggml_get_rows(ctx, weight("position_embd.weight"), g->positions), x);
        x = norm(ctx, x, "token_embd_norm");
        if (pooling == 2 || pooling == 3) {
            g->selected = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, sequences);
            ggml_set_input(g->selected);
        }
        for (int layer = 0; layer < layers; ++layer) {
            const auto p = "blk." + std::to_string(layer) + ".";
            auto * residual = x;
            ggml_tensor * q, * k, * v;
            if (find(p + "attn_qkv.weight")) {
                auto * qkv = linear(ctx, x, p + "attn_qkv");
                auto view = [&](int i) {
                    return ggml_view_4d(ctx, qkv, head, heads, length, batch,
                        head * sizeof(float), qkv->nb[1], length * qkv->nb[1], i * dim * sizeof(float));
                };
                q = view(0); k = view(1); v = view(2);
            } else {
                q = ggml_reshape_4d(ctx, linear(ctx, x, p + "attn_q"), head, heads, length, batch);
                k = ggml_reshape_4d(ctx, linear(ctx, x, p + "attn_k"), head, heads, length, batch);
                v = ggml_reshape_4d(ctx, linear(ctx, x, p + "attn_v"), head, heads, length, batch);
            }
            if (segments.empty()) x = attention(ctx, q, k, v, g->mask, length, batch);
            else {
                // Projections share one compact batch, while each CPU attention
                // retains the same shape/reduction order as individual inference.
                std::vector<ggml_tensor *> parts;
                int start = 0;
                for (size_t s = 0; s < segments.size(); ++s) {
                    auto slice = [&](ggml_tensor * value) {
                        return ggml_view_4d(ctx, value, head, heads, segments[s], 1,
                            value->nb[1], value->nb[2], value->nb[3], start * value->nb[2]);
                    };
                    parts.push_back(attention(ctx, slice(q), slice(k), slice(v), g->segment_masks[s], segments[s], 1));
                    start += segments[s];
                }
                while (parts.size() > 1) {
                    std::vector<ggml_tensor *> joined;
                    for (size_t i = 0; i < parts.size(); i += 2)
                        joined.push_back(i + 1 < parts.size() ? ggml_concat(ctx, parts[i], parts[i + 1], 1) : parts[i]);
                    parts = std::move(joined);
                }
                x = parts[0];
            }
            // CLS/last pooling never needs the final FFN for discarded token rows.
            if (layer == layers - 1 && g->selected) {
                x = ggml_get_rows(ctx, x, g->selected);
                residual = ggml_get_rows(ctx, residual, g->selected);
            }
            x = norm(ctx, ggml_add(ctx, linear(ctx, x, p + "attn_output"), residual), p + "attn_output_norm");
            residual = x;
            x = ggml_gelu(ctx, linear(ctx, x, p + "ffn_up"));
            x = norm(ctx, ggml_add(ctx, linear(ctx, x, p + "ffn_down"), residual), p + "layer_output_norm");
        }
        if (pooling == 1) {
            g->mean_weights = packed ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, length, sequences) :
                ggml_new_tensor_3d(ctx, GGML_TYPE_F32, length, 1, batch);
            ggml_set_input(g->mean_weights);
            auto * rows = packed ? x : ggml_reshape_3d(ctx, x, dim, length, batch);
            x = ggml_reshape_2d(ctx, ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, rows)), g->mean_weights), dim, sequences);
        }
        g->output = x;
        ggml_set_output(x); ggml_build_forward_expand(g->gf, x);
#if defined(TSG_GGML_USE_METAL)
        if (ggml_backend_is_metal(backend) && backend->iface.graph_optimize) {
            // Match the scheduler's fusion-preserving reorder before gallocr
            // plans tensor lifetimes. Metal currently adds no allocation edges.
            bool extra_dependency = false;
            ggml_backend_graph_optimize_params params = {
                [](void * state, ggml_tensor *, ggml_tensor *) { *static_cast<bool *>(state) = true; },
                &extra_dependency,
            };
            backend->iface.graph_optimize(backend, g->gf, &params);
            require(!extra_dependency, "Metal optimizer requested an unsupported allocation dependency");
        }
#endif
        for (int i = 0; i < ggml_graph_n_nodes(g->gf); ++i) {
            auto * node = ggml_graph_node(g->gf, i);
            require(ggml_backend_supports_op(precise_backend ? precise_backend : backend, node),
                "backend does not support " + std::string(ggml_op_name(node->op)));
        }
        g->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(g->allocator && ggml_gallocr_alloc_graph(g->allocator, g->gf), "cannot allocate inference graph");
        if (ggml_backend_is_cpu(backend)) {
            // Shape and thread count remain fixed while this graph is cached.
            // Retain its CPU work plan instead of replanning every HTTP request.
            g->plan_backend = backend;
            g->cpu_plan = ggml_backend_graph_plan_create(backend, g->gf);
            require(g->cpu_plan != nullptr, "cannot allocate CPU inference plan");
        }
        return g;
    }
    void encode(const int32_t * tokens, const int32_t * lengths, int batch, float * output, int capacity) {
        require(tokens && lengths && output && batch > 0 && batch <= 1024 && int64_t(batch) * dim <= capacity,
                "invalid batch or output buffer");
        int length = 0;
        int64_t total = 0;
        bool same_length = true;
        for (int i = 0; i < batch; ++i) {
            require(lengths[i] > 0 && lengths[i] <= context, "sequence exceeds model context");
            length = std::max(length, lengths[i]);
            total += lengths[i];
            same_length = same_length && lengths[i] == lengths[0];
        }
        require(total <= 65536, "microbatch exceeds 65536 tokens");
        const bool packed = batch > 1 && !same_length;
        std::vector<int> segments;
        if (packed && ggml_backend_is_cpu(backend)) {
            length = 0;
            for (int b = 0; b < batch; ++b) {
                int size = std::min(context, ((lengths[b] + cpu_alignment - 1) / cpu_alignment) * cpu_alignment);
                segments.push_back(size); length += size;
            }
        }
        const int streams = packed ? 1 : batch;
        if (packed && segments.empty()) length = int(total);
        // Align CPU columns for the quantized matrix kernels. Odd work chunks
        // switch between dot/MMLA accumulation and amplify rounding across layers.
        if (ggml_backend_is_cpu(backend) && segments.empty()) length = packed ? ((length + cpu_alignment - 1) / cpu_alignment) * cpu_alignment :
            std::min(context, ((length + cpu_alignment - 1) / cpu_alignment) * cpu_alignment);
        require(int64_t(length) * streams <= 65536, "microbatch exceeds 65536 padded tokens");
        // One context serializes graph reuse and backend submissions.
        std::lock_guard<std::mutex> lock(mutex);
        if (!cached || cached->length != length || cached->batch != streams || cached->sequences != batch || cached->packed != packed || cached->segments != segments) {
            cached.reset(); cached = build(length, streams, batch, packed, segments);
        }
        auto & g = *cached;
        std::vector<int32_t> padded(size_t(length) * streams, 0), positions(padded.size()), selected(batch);
        std::vector<ggml_fp16_t> mask(g.mask ? size_t(ggml_nelements(g.mask)) : 0, ggml_fp32_to_fp16(-INFINITY));
        std::vector<float> means(g.mean_weights ? size_t(ggml_nelements(g.mean_weights)) : 0, 0.0f);
        size_t input_offset = 0;
        size_t segment_start = 0;
        for (int b = 0; b < batch; ++b) {
            const size_t start = !segments.empty() ? segment_start : packed ? input_offset : size_t(b) * length;
            for (int i = 0; i < lengths[b]; ++i) {
                const int id = tokens[input_offset++];
                require(id >= 0 && id < vocab, "token id is outside model vocabulary");
                padded[start + i] = id;
                positions[start + i] = i;
                if (pooling == 1) means[size_t(b) * length + (packed ? start : 0) + i] = 1.0f / float(lengths[b]);
            }
            selected[b] = int(start) + (pooling == 3 ? lengths[b] - 1 : 0);
            if (!segments.empty()) {
                auto * tensor = g.segment_masks[b];
                std::vector<ggml_fp16_t> local_mask(size_t(ggml_nelements(tensor)), ggml_fp32_to_fp16(-INFINITY));
                for (int q = 0; q < tensor->ne[1]; ++q)
                    std::fill(local_mask.data() + size_t(q) * segments[b], local_mask.data() + size_t(q) * segments[b] + lengths[b], ggml_fp32_to_fp16(0.0f));
                ggml_backend_tensor_set(tensor, local_mask.data(), 0, local_mask.size() * sizeof(ggml_fp16_t));
                segment_start += segments[b];
            } else if (packed) {
                for (int q = 0; q < lengths[b]; ++q) {
                    auto * row = mask.data() + (start + q) * length + start;
                    std::fill(row, row + lengths[b], ggml_fp32_to_fp16(0.0f));
                }
            } else for (int q = 0; g.mask && q < g.mask->ne[1]; ++q) {
                auto * row = mask.data() + (size_t(b) * g.mask->ne[1] + q) * length;
                std::fill(row, row + lengths[b], ggml_fp32_to_fp16(0.0f));
            }
        }
        // Padded query rows must still have a finite softmax denominator. They
        // attend only to token zero and never participate in sentence pooling.
        if (packed && g.mask) for (size_t q = size_t(total); q < size_t(g.mask->ne[1]); ++q)
            mask[q * length] = ggml_fp32_to_fp16(0.0f);
        ggml_backend_tensor_set(g.tokens, padded.data(), 0, padded.size() * sizeof(int32_t));
        ggml_backend_tensor_set(g.positions, positions.data(), 0, positions.size() * sizeof(int32_t));
        if (g.mask) ggml_backend_tensor_set(g.mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        if (g.selected) ggml_backend_tensor_set(g.selected, selected.data(), 0, selected.size() * sizeof(int32_t));
        if (g.mean_weights) ggml_backend_tensor_set(g.mean_weights, means.data(), 0, means.size() * sizeof(float));
        const auto status = g.cpu_plan ? ggml_backend_graph_plan_compute(backend, g.cpu_plan) :
            ggml_backend_graph_compute(precise_backend ? precise_backend : backend, g.gf);
        require(status == GGML_STATUS_SUCCESS, "graph computation failed");
        ggml_backend_tensor_get(g.output, output, 0, size_t(batch) * dim * sizeof(float));
        for (int b = 0; b < batch; ++b) {
            auto * row = output + size_t(b) * dim;
            double sum = 0;
            for (int j = 0; j < dim; ++j) { require(std::isfinite(row[j]), "nonfinite output"); sum += double(row[j]) * row[j]; }
            require(sum > 0, "zero output vector");
            const double scale = 1.0 / std::sqrt(sum);
            for (int j = 0; j < dim; ++j) row[j] = float(double(row[j]) * scale);
        }
    }
};

std::unique_ptr<encoder> load(const char * path, const char * backend_name, int device, int threads) {
    require(path != nullptr && device >= 0, "invalid path or device");
    source_file source;
    source.file = gguf_init_from_file(path, {true, &source.tensors});
    require(source.file != nullptr, "cannot read model GGUF");
    auto * file = source.file;
    const auto arch_key = gguf_find_key(file, "general.architecture");
    require(arch_key >= 0 && gguf_get_kv_type(file, arch_key) == GGUF_TYPE_STRING &&
            std::string(gguf_get_val_str(file, arch_key)) == "bert", "supported encoder architecture is bert");
    auto e = std::make_unique<encoder>();
    e->dim = integer(file, "bert.embedding_length"); e->heads = integer(file, "bert.attention.head_count");
    e->layers = integer(file, "bert.block_count"); e->ff = integer(file, "bert.feed_forward_length");
    e->context = integer(file, "bert.context_length"); e->pooling = integer(file, "bert.pooling_type");
    e->eps = number(file, "bert.attention.layer_norm_epsilon", 1e-12f);
    auto * tok = ggml_get_tensor(source.tensors, "token_embd.weight");
    require(tok != nullptr && tok->ne[1] <= INT32_MAX, "missing or invalid token embeddings"); e->vocab = int(tok->ne[1]);
    require(e->dim > 0 && e->dim <= 16384 && e->heads > 0 && e->dim % e->heads == 0 && e->layers > 0 &&
            e->layers <= 128 && e->ff > 0 && e->ff <= 65536 && e->context > 0 && e->context <= 65536 &&
            std::isfinite(e->eps) && e->eps > 0 && e->pooling >= 1 && e->pooling <= 3, "invalid encoder metadata or pooling type");
    std::string name = backend_name ? backend_name : "CPU";
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::toupper(c)); });
    if (name.rfind("GGML_", 0) == 0) name = name.substr(5);
    if (name == "METAL") name = "MTL";
    if (name == "CPU") {
        require(device == 0, "CPU backend requires device zero");
        e->backend = ggml_backend_cpu_init();
        const int nthreads = threads > 0 ? threads : 4;
        require(nthreads <= 512, "thread count exceeds 512");
        if (e->backend) {
            ggml_backend_cpu_set_n_threads(e->backend, nthreads);
            auto params = ggml_threadpool_params_default(nthreads);
            e->threadpool = ggml_threadpool_new(&params);
            require(e->threadpool != nullptr, "cannot create CPU worker pool");
            ggml_backend_cpu_set_threadpool(e->backend, e->threadpool);
        }
    } else {
        int matched = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            auto reg = ggml_backend_dev_backend_reg(dev);
            std::string registered = reg ? ggml_backend_reg_name(reg) : "";
            std::transform(registered.begin(), registered.end(), registered.begin(), [](unsigned char c) { return char(std::toupper(c)); });
            if (name != registered) continue;
            if (matched++ == device) { e->backend = ggml_backend_dev_init(dev, nullptr); break; }
        }
    }
    require(e->backend != nullptr, "requested backend is unavailable: " + name);
#ifdef TSG_GGML_USE_CUDA
    // Prototype stays opt-in until both the numerical and balanced latency
    // gates pass. Do not change quantized weight storage or other backends.
    const char * q8_f32 = std::getenv("TS_EMBEDDING_Q8_F32");
    if (name == "CUDA" && q8_f32 && std::strcmp(q8_f32, "1") == 0) {
        e->precise_backend = tsg_dsv4_fused_backend_init(e->backend);
        require(e->precise_backend != nullptr, "cannot create Q8/F32 execution backend");
        std::fprintf(stderr, "Embedding encoder: TensorSharp Q8_0/F32 projections enabled (TS_EMBEDDING_Q8_F32=1).\n");
    }
#endif
    e->weights_ctx = ggml_init({size_t(gguf_get_n_tensors(file) + 1) * ggml_tensor_overhead() + 4096, nullptr, true});
    require(e->weights_ctx != nullptr, "cannot allocate weight metadata");
    std::map<std::string, std::vector<std::string>> fused;
    std::unordered_set<std::string> replaced;
    for (int layer = 0; layer < e->layers; ++layer) {
        const auto p = "blk." + std::to_string(layer) + ".";
        auto * q = ggml_get_tensor(source.tensors, (p + "attn_q.weight").c_str());
        auto * k = ggml_get_tensor(source.tensors, (p + "attn_k.weight").c_str());
        auto * v = ggml_get_tensor(source.tensors, (p + "attn_v.weight").c_str());
        auto * qb = ggml_get_tensor(source.tensors, (p + "attn_q.bias").c_str());
        auto * kb = ggml_get_tensor(source.tensors, (p + "attn_k.bias").c_str());
        auto * vb = ggml_get_tensor(source.tensors, (p + "attn_v.bias").c_str());
        const bool matching_bias = (!qb && !kb && !vb) || (qb && kb && vb && qb->type == kb->type && qb->type == vb->type &&
            ggml_nelements(qb) == e->dim && ggml_nelements(kb) == e->dim && ggml_nelements(vb) == e->dim);
        if (!q || !k || !v || q->type != k->type || q->type != v->type || !matching_bias ||
            q->ne[0] != e->dim || q->ne[1] != e->dim || q->ne[2] != 1 || q->ne[3] != 1 ||
            !ggml_are_same_shape(q, k) || !ggml_are_same_shape(q, v)) continue;
        // Concatenate rows once while loading, preserving the original GGUF quantization.
        // One QKV projection reduces kernel launches and reuses the input activation tile.
        for (const auto * suffix : {"weight", "bias"}) {
            if (std::strcmp(suffix, "bias") == 0 && !qb) continue;
            const auto name = p + "attn_qkv." + suffix;
            auto * tensor = std::strcmp(suffix, "weight") == 0 ? ggml_new_tensor_2d(e->weights_ctx, q->type, e->dim, 3 * e->dim) :
                ggml_new_tensor_1d(e->weights_ctx, qb->type, 3 * e->dim);
            ggml_set_name(tensor, name.c_str()); e->weights.emplace(name, tensor);
            auto & sources = fused[name];
            for (const auto * component : {"attn_q.", "attn_k.", "attn_v."}) {
                auto source_name = p + component + suffix;
                sources.push_back(source_name); replaced.insert(source_name);
            }
        }
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(file); ++i) {
        const std::string tensor_name = gguf_get_tensor_name(file, i);
        // Classification heads are not part of the sentence embedding graph.
        if (tensor_name.rfind("cls", 0) == 0) continue;
        if (replaced.count(tensor_name)) continue;
        auto * original = ggml_get_tensor(source.tensors, tensor_name.c_str());
        auto * tensor = ggml_dup_tensor(e->weights_ctx, original);
        ggml_set_name(tensor, tensor_name.c_str()); e->weights.emplace(tensor_name, tensor);
    }
    e->validate();
    std::unordered_set<std::string> packed_weights;
    if (ggml_backend_is_cpu(e->backend)) {
        auto * device = ggml_backend_get_device(e->backend);
        auto * registry = ggml_backend_dev_backend_reg(device);
        auto extra_types = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
            ggml_backend_reg_get_proc_address(registry, "ggml_backend_dev_get_extra_bufts"));
        auto ** candidates = extra_types ? extra_types(device) : nullptr;
        for (const auto & [name, tensor] : e->weights) {
            if (name.rfind("blk.", 0) != 0 || tensor->ne[1] <= 1) continue;
            for (auto ** candidate = candidates; candidate && *candidate; ++candidate) {
                auto * probe_ctx = ggml_init({8 * ggml_tensor_overhead() + 4096, nullptr, true});
                require(probe_ctx != nullptr, "cannot allocate CPU layout probe");
                auto * probe_buffer = ggml_backend_buft_alloc_buffer(*candidate, 0);
                if (!probe_buffer) { ggml_free(probe_ctx); continue; }
                auto * probe_weight = ggml_dup_tensor(probe_ctx, tensor);
                probe_weight->buffer = probe_buffer;
                auto * input = ggml_new_tensor_2d(probe_ctx, GGML_TYPE_F32, tensor->ne[0], 16);
                auto * op = ggml_mul_mat(probe_ctx, probe_weight, input);
                const bool supported = ggml_backend_dev_supports_op(device, op);
                ggml_backend_buffer_free(probe_buffer); ggml_free(probe_ctx);
                if (!supported) continue;
                auto * buffer = ggml_backend_buft_alloc_buffer(*candidate, ggml_backend_buft_get_alloc_size(*candidate, tensor));
                require(buffer != nullptr, "cannot allocate optimized CPU weights");
                e->optimized_buffers.push_back(buffer);
                require(ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS,
                    "cannot initialize optimized CPU weight tensor");
                ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                packed_weights.insert(name);
                break;
            }
        }
        bool all_quantized_optimized = !packed_weights.empty();
        for (const auto & [name, tensor] : e->weights)
            if (name.rfind("blk.", 0) == 0 && tensor->ne[1] > 1 && ggml_is_quantized(tensor->type) && !packed_weights.count(name))
                all_quantized_optimized = false;
        if (all_quantized_optimized) e->cpu_alignment = 4;
    }
    e->weights_buffer = ggml_backend_alloc_ctx_tensors(e->weights_ctx, e->backend);
    require(e->weights_buffer != nullptr, "cannot allocate model weights");
    ggml_backend_buffer_set_usage(e->weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::ifstream input(path, std::ios::binary);
    std::vector<char> staging(16 * 1024 * 1024);
    for (const auto & [name, tensor] : e->weights) {
        const auto parts = fused.count(name) ? fused.at(name) : std::vector<std::string>{name};
        // CPU repack buffers transform the complete matrix on upload, so fused
        // source pieces must be assembled before the single set_tensor call.
        std::vector<char> repack_data(packed_weights.count(name) ? ggml_nbytes(tensor) : 0);
        size_t destination_offset = 0;
        for (const auto & part : parts) {
            const auto i = gguf_find_tensor(file, part.c_str());
            auto * original = ggml_get_tensor(source.tensors, part.c_str());
            require(original && destination_offset <= ggml_nbytes(tensor) &&
                ggml_nbytes(original) <= ggml_nbytes(tensor) - destination_offset, "source exceeds tensor storage: " + part);
            input.seekg(gguf_get_data_offset(file) + gguf_get_tensor_offset(file, i));
            for (size_t offset = 0; offset < ggml_nbytes(original);) {
                const auto count = std::min(staging.size(), ggml_nbytes(original) - offset);
                require(bool(input.read(staging.data(), count)), "truncated tensor " + part);
                if (repack_data.empty()) ggml_backend_tensor_set(tensor, staging.data(), destination_offset + offset, count);
                else std::memcpy(repack_data.data() + destination_offset + offset, staging.data(), count);
                offset += count;
            }
            destination_offset += ggml_nbytes(original);
        }
        require(destination_offset == ggml_nbytes(tensor), "tensor storage size mismatch: " + name);
        if (!repack_data.empty()) ggml_backend_tensor_set(tensor, repack_data.data(), 0, repack_data.size());
    }
    return e;
}
}

TSG_EXPORT void * TSGgml_EmbeddingLoad(const char * path, const char * backend, int device, int threads) {
    try { return tsg_embeddings::load(path, backend, device, threads).release(); }
    catch (const std::exception & e) { tsg::set_last_error(e.what()); return nullptr; }
}
TSG_EXPORT void TSGgml_EmbeddingFree(void * handle) { delete static_cast<tsg_embeddings::encoder *>(handle); }
TSG_EXPORT int TSGgml_EmbeddingEncode(void * handle, const int32_t * tokens, const int32_t * lengths,
                                    int batch, float * output, int capacity) {
    try {
        tsg_embeddings::require(handle != nullptr, "null encoder");
        static_cast<tsg_embeddings::encoder *>(handle)->encode(tokens, lengths, batch, output, capacity);
        return 0;
    } catch (const std::exception & e) { tsg::set_last_error(e.what()); return -1; }
}
