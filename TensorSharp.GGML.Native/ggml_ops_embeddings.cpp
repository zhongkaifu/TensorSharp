// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_cuda_launch_policy.h"
#include "ggml_ops_internal.h"
#include "ggml_ops_graph_optimize.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "gguf.h"
#include "embedding_cpu_matmul.h"
#include "embedding_cpu_attention.h"
#include "embedding_cpu_profile.h"
#include "embedding_cpu_rms.h"

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
std::vector<int> integers(gguf_context * file, const std::string & name, int count, int fallback) {
    const auto key = gguf_find_key(file, name.c_str());
    if (key < 0 || gguf_get_kv_type(file, key) != GGUF_TYPE_ARRAY)
        return std::vector<int>(count, integer(file, name, fallback));
    require(gguf_get_arr_n(file, key) == size_t(count), "invalid array length for " + name);
    std::vector<int> result(count);
    const auto type = gguf_get_arr_type(file, key);
    const auto * data = gguf_get_arr_data(file, key);
    for (int i = 0; i < count; ++i) {
        if (type == GGUF_TYPE_UINT32) {
            const auto value = static_cast<const uint32_t *>(data)[i];
            require(value <= INT32_MAX, "invalid metadata " + name); result[i] = int(value);
        } else if (type == GGUF_TYPE_INT32) result[i] = static_cast<const int32_t *>(data)[i];
        else if (type == GGUF_TYPE_BOOL) result[i] = static_cast<const bool *>(data)[i] ? 1 : 0;
        else require(false, "invalid array type for " + name);
        require(result[i] >= 0, "invalid metadata " + name);
    }
    return result;
}
}

struct encoder {
    int dim = 0, output_dim = 0, heads = 0, layers = 0, ff = 0, context = 0, vocab = 0, pooling = 0;
    int full_head = 0, swa_head = 0, full_rot = 0, swa_rot = 0, swa_window = 0, per_layer_dim = 0;
    float full_rope_base = 10000.0f, swa_rope_base = 10000.0f, rope_scale = 1.0f;
    std::vector<int> kv_heads, sliding;
    float eps = 0;
    bool nomic_bert = false;
    bool gemma2 = false;
    bool cpu_float_linears = false;
    bool cpu_tiled_linears = false;
    bool cpu_flat_qkv_rms = false;
    std::unique_ptr<linux_blas_plan> cpu_blas;
    std::unique_ptr<cpu_attention_workspace> cpu_attention_scratch;
    ggml_backend_t backend = nullptr;
    ggml_threadpool_t threadpool = nullptr;
    ggml_context * weights_ctx = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    std::vector<ggml_backend_buffer_t> optimized_buffers;
    int cpu_alignment = 16;
    int gpu_key_alignment = 64;
    std::map<std::string, ggml_tensor *> weights;
    std::mutex mutex;
    struct graph {
        ggml_context * ctx = nullptr;
        ggml_context * allocation_ctx = nullptr;
        ggml_cgraph * allocation_graph = nullptr;
        ggml_gallocr_t allocator = nullptr;
        ggml_cgraph * gf = nullptr;
        ggml_backend_t plan_backend = nullptr;
        ggml_backend_graph_plan_t cpu_plan = nullptr;
        std::unique_ptr<cpu_profile_plan> cpu_profile;
        cpu_profile_plan::tensor_set attention_profile_nodes;
        ggml_tensor * tokens = nullptr, * embeddings = nullptr, * positions = nullptr, * mask = nullptr, * swa_mask = nullptr;
        ggml_tensor * selected = nullptr, * mean_weights = nullptr, * output = nullptr;
        int length = 0, batch = 0, sequences = 0;
        bool packed = false;
        std::vector<int> segments;
        std::vector<ggml_tensor *> segment_masks;
        std::vector<ggml_tensor *> segment_swa_masks;
        std::vector<std::unique_ptr<cpu_attention_node>> attention_nodes;
        ~graph() {
            cpu_profile.reset();
            if (cpu_plan) ggml_backend_graph_plan_free(plan_backend, cpu_plan);
            if (allocator) ggml_gallocr_free(allocator);
            if (allocation_ctx) ggml_free(allocation_ctx);
            if (ctx) ggml_free(ctx);
        }
    };
    std::unique_ptr<graph> cached;
    std::unique_ptr<graph> cached_rows;
    ~encoder() {
        cached.reset();
        cached_rows.reset();
        if (cpu_attention_scratch) std::fprintf(stderr, "EMBEDDING_CPU_ATTENTION tiles=%llu scalar_products=%llu scratch_bytes=%llu simd_softmax=%d\n",
            (unsigned long long)cpu_attention_scratch->tiles.load(),
            (unsigned long long)cpu_attention_scratch->scalar_products.load(),
            (unsigned long long)(cpu_attention_scratch->scores.size() * sizeof(float)), int(cpu_attention_scratch->simd_softmax));
        if (cpu_attention_scratch && cpu_attention_scratch->profile)
            std::fprintf(stderr, "EMBEDDING_CPU_ATTENTION_PROFILE diagnostic=1 accounting=sum-worker-intervals qk_ms=%.6f softmax_ms=%.6f pv_ms=%.6f\n",
                cpu_attention_scratch->qk_ns.load() / 1e6, cpu_attention_scratch->softmax_ns.load() / 1e6,
                cpu_attention_scratch->pv_ns.load() / 1e6);
        if (cpu_blas) {
            const auto counts = cpu_blas->counts();
            std::fprintf(stderr, "EMBEDDING_LINUX_BLAS available=%d library=%s reason=%s gemm_calls=%llu output_elements=%llu worker_mask=%llu policy_rejections=%llu shape_rejections=%llu\n",
                int(cpu_blas->available()), cpu_blas->library_path().c_str(), cpu_blas->reason().c_str(),
                (unsigned long long)counts.gemm_calls, (unsigned long long)counts.output_elements,
                (unsigned long long)counts.worker_mask, (unsigned long long)counts.policy_rejections,
                (unsigned long long)counts.shape_rejections);
        }
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
        // Quantized activation rounding is discontinuous. Changing just the PLE
        // projection's summation order can change later quantization bins, so
        // Accelerate is used only when every encoder linear is F32.
        // The measured Linux BLAS route regresses tiny projections; preserve
        // the ordinary precise CPU path below 32 token rows, including the
        // pooled output projection. Accelerate retains its existing policy.
        // On the qualified Linux x86 geometry, unchanged upstream F32 tiled
        // GEMM beats the optional BLAS policy on short/medium projections.
        // Larger batches keep BLAS; the much wider PLE matrix crosses sooner.
        // Select this only when the actual dependency build includes that GEMM.
        const bool tiled = cpu_tiled_linears && x->ne[1] <= 512 &&
            (w->ne[1] < 8192 || x->ne[1] <= 256);
        auto * blas = x->ne[1] >= 32 && !tiled ? cpu_blas.get() : nullptr;
        auto * y = gemma2 && ggml_backend_is_cpu(backend) && cpu_float_linears ? cpu_matmul(ctx, w, x, 1, blas) : nullptr;
        if (!y) y = ggml_mul_mat(ctx, w, x);
        if (auto * bias = find(name + ".bias")) y = ggml_add(ctx, y, bias);
        return y;
    }
    ggml_tensor * norm(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
        return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), weight(name + ".weight", dim, 1)),
                        weight(name + ".bias", dim, 1));
    }
    ggml_tensor * rms(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
        return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight(name + ".weight", x->ne[0], 1));
    }
    void validate() const {
        weight("token_embd.weight", dim, vocab);
        if (gemma2) {
            weight("per_layer_model_proj.weight", dim, int64_t(per_layer_dim) * layers);
            weight("per_layer_proj_norm.weight", per_layer_dim, 1);
            weight("output_norm.weight", dim, 1);
            weight("output.weight", dim, output_dim);
            for (int l = 0; l < layers; ++l) {
                const auto p = "blk." + std::to_string(l) + ".";
                const int head = sliding[l] ? swa_head : full_head;
                const int qdim = head * heads, kvdim = head * kv_heads[l];
                if (find(p + "attn_qkv.weight")) weight(p + "attn_qkv.weight", dim, qdim + 2 * kvdim);
                else {
                    weight(p + "attn_q.weight", dim, qdim);
                    weight(p + "attn_k.weight", dim, kvdim);
                    weight(p + "attn_v.weight", dim, kvdim);
                }
                weight(p + "attn_output.weight", qdim, dim);
                weight(p + "attn_q_norm.weight", head, 1); weight(p + "attn_k_norm.weight", head, 1);
                for (const auto * part : {"attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm", "post_norm"})
                    weight(p + part + ".weight", dim, 1);
                if (find(p + "ffn_gate_up.weight")) weight(p + "ffn_gate_up.weight", dim, 2 * ff);
                else {
                    weight(p + "ffn_gate.weight", dim, ff); weight(p + "ffn_up.weight", dim, ff);
                }
                weight(p + "ffn_down.weight", ff, dim);
                weight(p + "inp_gate.weight", dim, per_layer_dim);
                weight(p + "proj.weight", per_layer_dim, dim);
                weight(p + "layer_output_scale.weight", 1, 1);
            }
            return;
        }
        if (!nomic_bert) weight("position_embd.weight", dim, context);
        weight("token_embd_norm.weight", dim, 1); weight("token_embd_norm.bias", dim, 1);
        if (find("token_types.weight")) weight("token_types.weight", dim);
        auto projection = [&](const std::string & name, int input, int output) {
            weight(name + ".weight", input, output);
            if (find(name + ".bias")) weight(name + ".bias", output, 1);
        };
        int ffn_input = nomic_bert ? dim : dim;
        int ffn_output = nomic_bert ? dim : ff;
        for (int l = 0; l < layers; ++l) {
            const auto p = "blk." + std::to_string(l) + ".";
            if (find(p + "attn_qkv.weight")) projection(p + "attn_qkv", dim, 3 * dim);
            else for (const auto * part : {"attn_q", "attn_k", "attn_v"}) projection(p + part, dim, dim);
            projection(p + "attn_output", dim, dim);
            projection(p + "ffn_up", dim, ffn_output);
            if (nomic_bert) projection(p + "ffn_gate", dim, dim);
            projection(p + "ffn_down", ffn_output, dim);
            for (const auto * part : {"attn_output_norm", "layer_output_norm"}) {
                weight(p + part + ".weight", dim, 1); weight(p + part + ".bias", dim, 1);
            }
        }
    }
    ggml_tensor * attention(graph & g, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                            ggml_tensor * mask, int length, int batch, int local_radius = -1) const {
        auto * ctx = g.ctx;
        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        k = ggml_permute(ctx, k, 0, 2, 1, 3);
        v = ggml_permute(ctx, v, 0, 2, 1, 3);
        if (cpu_attention_scratch) {
            if (auto * tiled = cpu_attention(ctx, q, k, v, mask, local_radius, 1.0f, *cpu_attention_scratch, g.attention_nodes)) {
                g.attention_profile_nodes.insert(tiled);
                return ggml_reshape_2d(ctx, tiled, q->ne[0] * q->ne[2], length * batch);
            }
        }
        // Align only the key/value rows for GPU flash attention. Padding the
        // encoder's token rows multiplies all projections and FFNs for short
        // inputs; Q and the sentence states must retain their actual length.
        if (gemma2 && mask && mask->ne[0] > k->ne[1]) {
            const int64_t padding = mask->ne[0] - k->ne[1];
            k = ggml_pad(ctx, k, 0, padding, 0, 0);
            v = ggml_pad(ctx, v, 0, padding, 0, 0);
        }
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
        const int attention_dim = int(q->ne[0] * q->ne[2]);
        const float scale = gemma2 ? 1.0f : 1.0f / std::sqrt(float(q->ne[0]));
        auto * fa = ggml_flash_attn_ext(ctx, q, kh, vh, mask, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(fa, GGML_PREC_F32);
        if (ggml_backend_supports_op(backend, fa)) return ggml_reshape_2d(ctx, fa, attention_dim, length * batch);
        auto * scores = ggml_mul_mat(ctx, k, q);
        g.attention_profile_nodes.insert(scores);
        auto * mask_f32 = mask ? ggml_cast(ctx, mask, GGML_TYPE_F32) : nullptr;
        scores = ggml_soft_max_ext(ctx, scores, mask_f32, scale, 0.0f);
        g.attention_profile_nodes.insert(scores);
        auto * out = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, v)), scores);
        g.attention_profile_nodes.insert(out);
        return ggml_cont_2d(ctx, ggml_permute(ctx, out, 0, 2, 1, 3), attention_dim, length * batch);
    }
    ggml_tensor * gemma_layers(graph & g, ggml_tensor * x, int length, int batch,
                               const std::vector<int> & segments) const {
        auto * ctx = g.ctx;
        const int n = length * batch;
        if (!g.embeddings) x = ggml_scale(ctx, x, std::sqrt(float(dim)));
        auto * per_layer = linear(ctx, x, "per_layer_model_proj");
        per_layer = ggml_scale(ctx, per_layer, 1.0f / std::sqrt(float(dim)));
        per_layer = ggml_reshape_3d(ctx, per_layer, per_layer_dim, layers, n);
        per_layer = rms(ctx, per_layer, "per_layer_proj_norm");
        per_layer = ggml_cont(ctx, ggml_permute(ctx, per_layer, 0, 2, 1, 3));
        for (int layer = 0; layer < layers; ++layer) {
            const auto p = "blk." + std::to_string(layer) + ".";
            const bool swa = sliding[layer] != 0;
            const int head = swa ? swa_head : full_head, kv = kv_heads[layer];
            const int qdim = head * heads, kvdim = head * kv;
            auto * residual = x;
            auto * normalized = rms(ctx, x, p + "attn_norm");
            ggml_tensor * q, * k, * v;
            if (find(p + "attn_qkv.weight")) {
                auto * qkv = linear(ctx, normalized, p + "attn_qkv");
                auto view = [&](int nheads, int offset) {
                    return ggml_view_3d(ctx, qkv, head, nheads, n,
                        head * sizeof(float), qkv->nb[1], offset * sizeof(float));
                };
                q = view(heads, 0); k = view(kv, qdim); v = view(kv, qdim + kvdim);
            } else {
                q = ggml_reshape_3d(ctx, linear(ctx, normalized, p + "attn_q"), head, heads, n);
                k = ggml_reshape_3d(ctx, linear(ctx, normalized, p + "attn_k"), head, kv, n);
                v = ggml_reshape_3d(ctx, linear(ctx, normalized, p + "attn_v"), head, kv, n);
            }
            auto qkv_rms = [&](ggml_tensor * value, ggml_tensor * norm_weight) {
                auto * result = cpu_flat_qkv_rms ? cpu_qkv_rms(ctx, value, norm_weight, &eps) : nullptr;
                if (result) return result;
                result = ggml_rms_norm(ctx, value, eps);
                return norm_weight ? ggml_mul(ctx, result, norm_weight) : result;
            };
            q = qkv_rms(q, weight(p + "attn_q_norm.weight", head, 1));
            k = qkv_rms(k, weight(p + "attn_k_norm.weight", head, 1));
            v = qkv_rms(v, nullptr);
            const float base = swa ? swa_rope_base : full_rope_base;
            const int rot = swa ? swa_rot : full_rot;
            q = ggml_rope_ext(ctx, q, g.positions, nullptr, rot, GGML_ROPE_TYPE_NEOX,
                0, base, rope_scale, 0.0f, 1.0f, 0.0f, 0.0f);
            k = ggml_rope_ext(ctx, k, g.positions, nullptr, rot, GGML_ROPE_TYPE_NEOX,
                0, base, rope_scale, 0.0f, 1.0f, 0.0f, 0.0f);
            q = ggml_reshape_4d(ctx, q, head, heads, length, batch);
            k = ggml_reshape_4d(ctx, k, head, kv, length, batch);
            v = ggml_reshape_4d(ctx, v, head, kv, length, batch);
            if (segments.empty()) x = attention(g, q, k, v, swa ? g.swa_mask : g.mask, length, batch, swa ? swa_window / 2 : -1);
            else {
                std::vector<ggml_tensor *> parts;
                int start = 0;
                for (size_t s = 0; s < segments.size(); ++s) {
                    auto slice = [&](ggml_tensor * value) {
                        return ggml_view_4d(ctx, value, head, value->ne[1], segments[s], 1,
                            value->nb[1], value->nb[2], value->nb[3], start * value->nb[2]);
                    };
                    auto * mask = swa ? g.segment_swa_masks[s] : g.segment_masks[s];
                    parts.push_back(attention(g, slice(q), slice(k), slice(v), mask, segments[s], 1, swa ? swa_window / 2 : -1));
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
            x = linear(ctx, x, p + "attn_output");
            if (layer == layers - 1 && g.selected) {
                x = ggml_get_rows(ctx, x, g.selected);
                residual = ggml_get_rows(ctx, residual, g.selected);
            }
            x = ggml_add(ctx, rms(ctx, x, p + "post_attention_norm"), residual);
            residual = x;
            normalized = rms(ctx, x, p + "ffn_norm");
            if (find(p + "ffn_gate_up.weight")) {
                auto * gate_up = linear(ctx, normalized, p + "ffn_gate_up");
                auto * gate = ggml_view_2d(ctx, gate_up, ff, gate_up->ne[1], gate_up->nb[1], 0);
                auto * up = ggml_view_2d(ctx, gate_up, ff, gate_up->ne[1], gate_up->nb[1], ff * sizeof(float));
                x = ggml_geglu_split(ctx, gate, up);
            } else {
                auto * gate = linear(ctx, normalized, p + "ffn_gate");
                auto * up = linear(ctx, normalized, p + "ffn_up");
                x = ggml_geglu_split(ctx, gate, up);
            }
            x = ggml_add(ctx, rms(ctx, linear(ctx, x, p + "ffn_down"), p + "post_ffw_norm"), residual);
            residual = x;
            auto * layer_input = ggml_view_2d(ctx, per_layer, per_layer_dim, n, per_layer->nb[1], layer * per_layer->nb[2]);
            if (layer == layers - 1 && g.selected) layer_input = ggml_get_rows(ctx, layer_input, g.selected);
            x = ggml_mul(ctx, ggml_gelu(ctx, linear(ctx, x, p + "inp_gate")), layer_input);
            x = ggml_add(ctx, rms(ctx, linear(ctx, x, p + "proj"), p + "post_norm"), residual);
            x = ggml_mul(ctx, x, weight(p + "layer_output_scale.weight", 1, 1));
        }
        return rms(ctx, x, "output_norm");
    }
    std::unique_ptr<graph> build(int length, int batch, int sequences, bool packed, const std::vector<int> & segments,
                                 bool external_embeddings = false) const {
        auto g = std::make_unique<graph>();
        g->length = length; g->batch = batch; g->sequences = sequences; g->packed = packed;
        g->segments = segments;
        const int n = length * batch, head = dim / heads;
        const size_t nodes = size_t(layers) * (160 + 24 * segments.size()) + 256;
        g->ctx = ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false) + 4096, nullptr, true});
        require(g->ctx != nullptr, "cannot allocate graph metadata");
        auto * ctx = g->ctx;
        g->gf = ggml_new_graph_custom(ctx, nodes, false);
        if (external_embeddings) {
            g->embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, n); ggml_set_input(g->embeddings);
        } else {
            g->tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n); ggml_set_input(g->tokens);
        }
        g->positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_set_input(g->positions);
        // Equal lengths use independent streams; packed inputs use a block mask.
        // Mask rows are padded for the flash-attention kernels on CUDA and Metal.
        const int mask_queries = ((length + 31) / 32) * 32;
        if (segments.empty() && (gemma2 || packed || ggml_backend_is_cpu(backend))) {
            const int mask_keys = gemma2 && !ggml_backend_is_cpu(backend) ?
                ((length + gpu_key_alignment - 1) / gpu_key_alignment) * gpu_key_alignment : length;
            g->mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, mask_keys, mask_queries, 1, batch);
            ggml_set_input(g->mask);
            if (gemma2) {
                g->swa_mask = ggml_dup_tensor(ctx, g->mask); ggml_set_input(g->swa_mask);
            }
        }
        for (int size : segments) {
            const int keys = gemma2 && !ggml_backend_is_cpu(backend) ?
                ((size + gpu_key_alignment - 1) / gpu_key_alignment) * gpu_key_alignment : size;
            auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, keys, ((size + 31) / 32) * 32);
            ggml_set_input(mask); g->segment_masks.push_back(mask);
            if (gemma2) {
                auto * swa = ggml_dup_tensor(ctx, mask); ggml_set_input(swa); g->segment_swa_masks.push_back(swa);
            }
        }
        auto * x = g->embeddings ? g->embeddings : ggml_get_rows(ctx, weight("token_embd.weight"), g->tokens);
        if (!gemma2) {
            if (auto * types = find("token_types.weight")) {
                auto * first_type = ggml_view_1d(ctx, types, dim, 0);
                if (first_type->type != GGML_TYPE_F32) first_type = ggml_cast(ctx, first_type, GGML_TYPE_F32);
                x = ggml_add(ctx, x, first_type);
            }
            if (!nomic_bert) x = ggml_add(ctx, ggml_get_rows(ctx, weight("position_embd.weight"), g->positions), x);
            x = norm(ctx, x, "token_embd_norm");
        }
        if (pooling == 2 || pooling == 3) {
            g->selected = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, sequences);
            ggml_set_input(g->selected);
        }
        if (gemma2) x = gemma_layers(*g, x, length, batch, segments);
        else for (int layer = 0; layer < layers; ++layer) {
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
            if (nomic_bert) {
                // RoPE: rotate Q and K pairs within each head by position-dependent angles.
                // rope_type=2 (NeoX-style) rotates pairs [0,1], [2,3], ... within each head.
                auto * pos_4d = ggml_reshape_4d(ctx, g->positions, 1, 1, length, batch);
                q = ggml_rope_ext(ctx, q, pos_4d, nullptr, head, 2, 0, 1000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                k = ggml_rope_ext(ctx, k, pos_4d, nullptr, head, 2, 0, 1000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                // RoPE must be done before attention; if the graph optimizer fused RoPE
                // with the downstream cast/permute the backend must still support it.
            }
            if (segments.empty()) x = attention(*g, q, k, v, g->mask, length, batch);
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
                    parts.push_back(attention(*g, slice(q), slice(k), slice(v), g->segment_masks[s], segments[s], 1));
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
            if (nomic_bert) {
                auto * up = linear(ctx, x, p + "ffn_up");
                auto * gate = linear(ctx, x, p + "ffn_gate");
                x = ggml_mul(ctx, ggml_gelu(ctx, gate), up);
            } else {
                x = ggml_gelu(ctx, linear(ctx, x, p + "ffn_up"));
            }
            x = norm(ctx, ggml_add(ctx, linear(ctx, x, p + "ffn_down"), residual), p + "layer_output_norm");
        }
        if (pooling == 1) {
            g->mean_weights = packed ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, length, sequences) :
                ggml_new_tensor_3d(ctx, GGML_TYPE_F32, length, 1, batch);
            ggml_set_input(g->mean_weights);
            auto * rows = packed ? x : ggml_reshape_3d(ctx, x, dim, length, batch);
            x = ggml_reshape_2d(ctx, ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, rows)), g->mean_weights), dim, sequences);
        }
        // The final linear projection commutes with mean pooling. Projecting
        // one sentence vector avoids the per-token 512 x 768 multiplication.
        if (gemma2) x = linear(ctx, x, "output");
        g->output = x;
        ggml_set_output(x); ggml_build_forward_expand(g->gf, x);
#if defined(TSG_GGML_USE_METAL)
        if (ggml_backend_is_metal(backend) && backend->iface.graph_optimize) {
            // Preserve external inputs until the last node of each fused Metal
            // operation, exactly as ggml's scheduler does. The dependency views
            // belong only to the allocation graph, so compute fusion stays intact.
            tsg_alloc_dependencies dependencies;
            tsg_collect_graph_optimize_dependencies(backend, g->gf, dependencies);
            if (!dependencies.empty()) {
                size_t count = 0;
                for (const auto & item : dependencies) count += (item.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
                const size_t capacity = size_t(g->gf->n_nodes) + count;
                g->allocation_ctx = ggml_init({count * ggml_tensor_overhead() +
                    ggml_graph_overhead_custom(capacity, false) + 4096, nullptr, true});
                require(g->allocation_ctx != nullptr, "cannot allocate Metal fusion dependencies");
                g->allocation_graph = ggml_new_graph_custom(g->allocation_ctx, capacity, false);
                auto * copy = g->allocation_graph;
                for (int i = 0; i < g->gf->n_leafs; ++i) copy->leafs[copy->n_leafs++] = g->gf->leafs[i];
                size_t added = 0;
                for (int i = 0; i < g->gf->n_nodes; ++i) {
                    auto * node = g->gf->nodes[i]; copy->nodes[copy->n_nodes++] = node;
                    const auto found = dependencies.find(node);
                    if (found == dependencies.end()) continue;
                    const auto & keep = found->second;
                    for (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                        auto * dep = ggml_view_tensor(g->allocation_ctx, keep[k]);
                        for (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); ++s) dep->src[s] = keep[k + s];
                        copy->nodes[copy->n_nodes++] = dep; ++added;
                    }
                }
                require(added == count, "Metal fusion dependency references a missing graph node");
            }
        }
#endif
        for (int i = 0; i < ggml_graph_n_nodes(g->gf); ++i) {
            auto * node = ggml_graph_node(g->gf, i);
            require(ggml_backend_supports_op(backend, node),
                "backend does not support " + std::string(ggml_op_name(node->op)));
        }
        g->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(g->allocator && ggml_gallocr_alloc_graph(g->allocator, g->allocation_graph ? g->allocation_graph : g->gf),
            "cannot allocate inference graph");
        if (ggml_backend_is_cpu(backend)) {
            // Shape and thread count remain fixed while this graph is cached.
            // Retain its CPU work plan instead of replanning every HTTP request.
            g->plan_backend = backend;
            g->cpu_plan = ggml_backend_graph_plan_create(backend, g->gf);
            require(g->cpu_plan != nullptr, "cannot allocate CPU inference plan");
            if (cpu_profile_plan::enabled()) {
                cpu_profile_plan::tensor_set learned_weights;
                for (const auto & item : weights) learned_weights.insert(item.second);
                g->cpu_profile = std::make_unique<cpu_profile_plan>(backend, g->gf, learned_weights, g->attention_profile_nodes);
                require(g->cpu_profile->valid(), "cannot allocate diagnostic CPU profile plans");
            }
        }
        return g;
    }
    void encode(const int32_t * tokens, const int32_t * lengths, int batch, float * output, int capacity,
                const float * embeddings = nullptr, int input_capacity = 0) {
        require((tokens || (gemma2 && embeddings)) && lengths && output && batch > 0 && batch <= 1024 && int64_t(batch) * output_dim <= capacity,
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
        if (embeddings) require(total * dim <= input_capacity, "input embedding buffer is too small");
        const bool packed = batch > 1 && (!same_length ||
            (gemma2 && !ggml_backend_is_cpu(backend) && total <= 1024));
        std::vector<int> segments;
        // Short GPU batches share one block-diagonal attention invocation.
        // Larger mixed batches keep compact independent attention segments to
        // avoid allocating quadratic masks over the entire batch.
        if (packed && (ggml_backend_is_cpu(backend) || (gemma2 && total > 1024))) {
            length = 0;
            for (int b = 0; b < batch; ++b) {
                const int alignment = ggml_backend_is_cpu(backend) ? cpu_alignment : 1;
                int size = std::min(context, ((lengths[b] + alignment - 1) / alignment) * alignment);
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
        if (!cached || cached->length != length || cached->batch != streams || cached->sequences != batch || cached->packed != packed || cached->segments != segments ||
            bool(cached->embeddings) != bool(embeddings)) {
            cached.reset(); cached = build(length, streams, batch, packed, segments, embeddings != nullptr);
        }
        auto & g = *cached;
        std::vector<int32_t> padded(size_t(length) * streams, 0), positions(padded.size()), selected(batch);
        std::vector<float> padded_embeddings(embeddings ? padded.size() * dim : 0, 0.0f);
        std::vector<ggml_fp16_t> mask(g.mask ? size_t(ggml_nelements(g.mask)) : 0, ggml_fp32_to_fp16(-INFINITY));
        std::vector<ggml_fp16_t> swa_mask(g.swa_mask ? size_t(ggml_nelements(g.swa_mask)) : 0, ggml_fp32_to_fp16(-INFINITY));
        std::vector<float> means(g.mean_weights ? size_t(ggml_nelements(g.mean_weights)) : 0, 0.0f);
        size_t input_offset = 0;
        size_t segment_start = 0;
        for (int b = 0; b < batch; ++b) {
            const size_t start = !segments.empty() ? segment_start : packed ? input_offset : size_t(b) * length;
            for (int i = 0; i < lengths[b]; ++i) {
                if (embeddings) {
                    auto * row = padded_embeddings.data() + (start + i) * dim;
                    const auto * source = embeddings + input_offset * dim;
                    for (int d = 0; d < dim; ++d) { require(std::isfinite(source[d]), "nonfinite input embedding"); row[d] = source[d]; }
                } else {
                    const int id = tokens[input_offset];
                    require(id >= 0 && id < vocab, "token id is outside model vocabulary");
                    padded[start + i] = id;
                }
                ++input_offset;
                positions[start + i] = i;
                if (pooling == 1) means[size_t(b) * length + (packed ? start : 0) + i] = 1.0f / float(lengths[b]);
            }
            selected[b] = int(start) + (pooling == 3 ? lengths[b] - 1 : 0);
            if (!segments.empty()) {
                auto * tensor = g.segment_masks[b];
                std::vector<ggml_fp16_t> local_mask(size_t(ggml_nelements(tensor)), ggml_fp32_to_fp16(-INFINITY));
                for (int q = 0; q < tensor->ne[1]; ++q)
                    std::fill(local_mask.data() + size_t(q) * tensor->ne[0], local_mask.data() + size_t(q) * tensor->ne[0] + lengths[b], ggml_fp32_to_fp16(0.0f));
                ggml_backend_tensor_set(tensor, local_mask.data(), 0, local_mask.size() * sizeof(ggml_fp16_t));
                if (gemma2) {
                    std::fill(local_mask.begin(), local_mask.end(), ggml_fp32_to_fp16(-INFINITY));
                    for (int q = 0; q < tensor->ne[1]; ++q) {
                        const int begin = std::max(0, q - swa_window / 2);
                        const int end = std::min(lengths[b], q + swa_window / 2 + 1);
                        auto * row = local_mask.data() + size_t(q) * tensor->ne[0];
                        if (begin < end) std::fill(row + begin, row + end, ggml_fp32_to_fp16(0.0f));
                        else row[0] = ggml_fp32_to_fp16(0.0f);
                    }
                    ggml_backend_tensor_set(g.segment_swa_masks[b], local_mask.data(), 0, local_mask.size() * sizeof(ggml_fp16_t));
                }
                segment_start += segments[b];
            } else if (packed) {
                for (int q = 0; q < lengths[b]; ++q) {
                    auto * row = mask.data() + (start + q) * g.mask->ne[0] + start;
                    std::fill(row, row + lengths[b], ggml_fp32_to_fp16(0.0f));
                    if (gemma2) {
                        const int begin = std::max(0, q - swa_window / 2);
                        const int end = std::min(lengths[b], q + swa_window / 2 + 1);
                        auto * swa_row = swa_mask.data() + (start + q) * g.swa_mask->ne[0] + start;
                        std::fill(swa_row + begin, swa_row + end, ggml_fp32_to_fp16(0.0f));
                    }
                }
            } else for (int q = 0; g.mask && q < g.mask->ne[1]; ++q) {
                auto * row = mask.data() + (size_t(b) * g.mask->ne[1] + q) * g.mask->ne[0];
                std::fill(row, row + lengths[b], ggml_fp32_to_fp16(0.0f));
                if (gemma2) {
                    auto * swa_row = swa_mask.data() + (size_t(b) * g.mask->ne[1] + q) * g.swa_mask->ne[0];
                    const int begin = std::max(0, q - swa_window / 2);
                    const int end = std::min(lengths[b], q + swa_window / 2 + 1);
                    if (begin < end) std::fill(swa_row + begin, swa_row + end, ggml_fp32_to_fp16(0.0f));
                    else swa_row[0] = ggml_fp32_to_fp16(0.0f);
                }
            }
        }
        // Padded query rows must still have a finite softmax denominator. They
        // attend only to token zero and never participate in sentence pooling.
        if (packed && g.mask) for (size_t q = size_t(total); q < size_t(g.mask->ne[1]); ++q) {
            mask[q * g.mask->ne[0]] = ggml_fp32_to_fp16(0.0f);
            if (g.swa_mask) swa_mask[q * g.swa_mask->ne[0]] = ggml_fp32_to_fp16(0.0f);
        }
        if (g.tokens) ggml_backend_tensor_set(g.tokens, padded.data(), 0, padded.size() * sizeof(int32_t));
        if (g.embeddings) ggml_backend_tensor_set(g.embeddings, padded_embeddings.data(), 0, padded_embeddings.size() * sizeof(float));
        ggml_backend_tensor_set(g.positions, positions.data(), 0, positions.size() * sizeof(int32_t));
        if (g.mask) ggml_backend_tensor_set(g.mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        if (g.swa_mask) ggml_backend_tensor_set(g.swa_mask, swa_mask.data(), 0, swa_mask.size() * sizeof(ggml_fp16_t));
        if (g.selected) ggml_backend_tensor_set(g.selected, selected.data(), 0, selected.size() * sizeof(int32_t));
        if (g.mean_weights) ggml_backend_tensor_set(g.mean_weights, means.data(), 0, means.size() * sizeof(float));
        const auto status = g.cpu_profile ? g.cpu_profile->compute(g.length, g.batch, g.sequences) :
            g.cpu_plan ? ggml_backend_graph_plan_compute(backend, g.cpu_plan) :
            ggml_backend_graph_compute(backend, g.gf);
        require(status == GGML_STATUS_SUCCESS, "graph computation failed");
        ggml_backend_tensor_get(g.output, output, 0, size_t(batch) * output_dim * sizeof(float));
        for (int b = 0; b < batch; ++b) {
            auto * row = output + size_t(b) * output_dim;
            double sum = 0;
            for (int j = 0; j < output_dim; ++j) { require(std::isfinite(row[j]), "nonfinite output"); sum += double(row[j]) * row[j]; }
            require(sum > 0, "zero output vector");
            const double scale = 1.0 / std::sqrt(sum);
            for (int j = 0; j < output_dim; ++j) row[j] = float(double(row[j]) * scale);
        }
    }
    void token_rows(const int32_t * tokens, int count, float * output, int capacity) {
        require(gemma2 && tokens && output && count > 0 && count <= 65536 && int64_t(count) * dim <= capacity,
            "invalid token embedding request");
        for (int i = 0; i < count; ++i) require(tokens[i] >= 0 && tokens[i] < vocab, "token id is outside model vocabulary");
        std::lock_guard<std::mutex> lock(mutex);
        if (!cached_rows || cached_rows->length != count) {
            cached_rows.reset();
            auto g = std::make_unique<graph>(); g->length = count;
            g->ctx = ggml_init({16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false) + 4096, nullptr, true});
            require(g->ctx != nullptr, "cannot allocate token embedding graph");
            g->gf = ggml_new_graph_custom(g->ctx, 16, false);
            g->tokens = ggml_new_tensor_1d(g->ctx, GGML_TYPE_I32, count); ggml_set_input(g->tokens);
            g->output = ggml_scale(g->ctx, ggml_get_rows(g->ctx, weight("token_embd.weight"), g->tokens), std::sqrt(float(dim)));
            ggml_set_output(g->output); ggml_build_forward_expand(g->gf, g->output);
            g->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            require(g->allocator && ggml_gallocr_alloc_graph(g->allocator, g->gf), "cannot allocate token embedding output");
            cached_rows = std::move(g);
        }
        auto & g = *cached_rows;
        ggml_backend_tensor_set(g.tokens, tokens, 0, size_t(count) * sizeof(int32_t));
        require(ggml_backend_graph_compute(backend, g.gf) == GGML_STATUS_SUCCESS, "token embedding computation failed");
        ggml_backend_tensor_get(g.output, output, 0, size_t(count) * dim * sizeof(float));
    }
};

std::unique_ptr<encoder> load(const char * path, const char * backend_name, int device, int threads) {
    require(path != nullptr && device >= 0, "invalid path or device");
    source_file source;
    source.file = gguf_init_from_file(path, {true, &source.tensors});
    require(source.file != nullptr, "cannot read model GGUF");
    auto * file = source.file;
    const auto arch_key = gguf_find_key(file, "general.architecture");
    require(arch_key >= 0 && gguf_get_kv_type(file, arch_key) == GGUF_TYPE_STRING, "missing architecture key");
    std::string arch = gguf_get_val_str(file, arch_key);
    bool is_nomic = arch == "nomic-bert";
    bool is_gemma2 = arch == "gemma-embedding2";
    require(arch == "bert" || is_nomic || is_gemma2, "supported encoder architectures are bert, nomic-bert, and gemma-embedding2");
    std::string prefix = arch + ".";
    auto e = std::make_unique<encoder>();
    e->nomic_bert = is_nomic;
    e->gemma2 = is_gemma2;
    e->dim = integer(file, prefix + "embedding_length"); e->heads = integer(file, prefix + "attention.head_count");
    e->layers = integer(file, prefix + "block_count"); e->ff = integer(file, prefix + "feed_forward_length");
    e->context = integer(file, prefix + "context_length"); e->pooling = integer(file, prefix + "pooling_type");
    require(e->dim > 0 && e->dim <= 16384 && e->heads > 0 && e->heads <= 512 && e->dim % e->heads == 0 &&
        e->layers > 0 && e->layers <= 128 && e->ff > 0 && e->ff <= 65536, "invalid encoder dimensions");
    e->eps = number(file, prefix + "attention.layer_norm_epsilon", 1e-12f);
    e->output_dim = e->dim;
    if (is_gemma2) {
        // The GGUF rotary range is 262144; the released embedding model was
        // trained and evaluated with a shared text/media window of 8192.
        e->context = std::min(e->context, 8192);
        e->output_dim = integer(file, prefix + "embedding_length_out", e->dim);
        e->eps = number(file, prefix + "attention.layer_norm_rms_epsilon", 1e-6f);
        e->per_layer_dim = integer(file, prefix + "embedding_length_per_layer_input");
        e->full_head = integer(file, prefix + "attention.key_length", e->dim / std::max(1, e->heads));
        e->swa_head = integer(file, prefix + "attention.key_length_swa", e->full_head);
        require(integer(file, prefix + "attention.value_length", e->full_head) == e->full_head &&
            integer(file, prefix + "attention.value_length_swa", e->swa_head) == e->swa_head,
            "EmbeddingGemma2 requires equal key and value head dimensions");
        e->full_rot = integer(file, prefix + "rope.dimension_count", e->full_head);
        e->swa_rot = integer(file, prefix + "rope.dimension_count_swa", e->full_rot);
        e->full_rope_base = number(file, prefix + "rope.freq_base", 1000000.0f);
        e->swa_rope_base = number(file, prefix + "rope.freq_base_swa", 10000.0f);
        const float rope_factor = number(file, prefix + "rope.scaling.factor", 1.0f);
        require(std::isfinite(rope_factor) && rope_factor > 0, "invalid rotary scaling factor");
        e->rope_scale = 1.0f / rope_factor;
        e->swa_window = integer(file, prefix + "attention.sliding_window");
        e->kv_heads = integers(file, prefix + "attention.head_count_kv", e->layers, e->heads);
        const auto pattern = gguf_find_key(file, (prefix + "attention.sliding_window_pattern").c_str());
        if (pattern >= 0 && gguf_get_kv_type(file, pattern) == GGUF_TYPE_ARRAY)
            e->sliding = integers(file, prefix + "attention.sliding_window_pattern", e->layers, 1);
        else {
            const int period = integer(file, prefix + "attention.sliding_window_pattern", 6);
            for (int l = 0; l < e->layers; ++l) e->sliding.push_back(period == 0 || l % period < period - 1);
        }
        require(e->output_dim > 0 && e->output_dim <= 16384 && e->per_layer_dim > 0 && e->per_layer_dim <= 16384 &&
            e->full_head > 0 && e->full_head <= 2048 && e->swa_head > 0 && e->swa_head <= 2048 &&
            e->full_rot > 0 && e->full_rot <= e->full_head && e->full_rot % 2 == 0 &&
            e->swa_rot > 0 && e->swa_rot <= e->swa_head && e->swa_rot % 2 == 0 &&
            e->swa_window > 0 && e->swa_window <= 65536 &&
            std::isfinite(e->full_rope_base) && e->full_rope_base > 0 &&
            std::isfinite(e->swa_rope_base) && e->swa_rope_base > 0, "invalid EmbeddingGemma2 metadata");
        for (int l = 0; l < e->layers; ++l) require(e->kv_heads[l] > 0 && e->heads % e->kv_heads[l] == 0 &&
            e->sliding[l] <= 1, "invalid EmbeddingGemma2 attention heads or sliding pattern");
    }
    auto * tok = ggml_get_tensor(source.tensors, "token_embd.weight");
    require(tok != nullptr && tok->ne[1] <= INT32_MAX, "missing or invalid token embeddings"); e->vocab = int(tok->ne[1]);
    require(e->dim > 0 && e->dim <= 16384 && e->heads > 0 && e->dim % e->heads == 0 && e->layers > 0 &&
            e->layers <= 128 && e->ff > 0 && e->ff <= 65536 && e->context > 0 && e->context <= 65536 &&
            std::isfinite(e->eps) && e->eps > 0 && e->pooling >= 1 && e->pooling <= 3, "invalid encoder metadata or pooling type");
    std::string name = backend_name ? backend_name : "CPU";
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::toupper(c)); });
    if (name.rfind("GGML_", 0) == 0) name = name.substr(5);
    if (name == "METAL") name = "MTL";
    if (name == "CUDA") e->gpu_key_alignment = 256;
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
            const auto * blas_setting = std::getenv("TSG_EMBEDDING_LINUX_BLAS");
            if (is_gemma2 && blas_setting && std::strcmp(blas_setting, "1") == 0)
                e->cpu_blas = std::make_unique<linux_blas_plan>(std::getenv("TSG_EMBEDDING_BLAS_LIBRARY"), std::min(nthreads, 8));
            const auto * attention_setting = std::getenv("TSG_EMBEDDING_CPU_ATTENTION");
            if (e->cpu_blas && e->cpu_blas->available() && attention_setting && std::strcmp(attention_setting, "1") == 0)
                e->cpu_attention_scratch = std::make_unique<cpu_attention_workspace>(e->cpu_blas.get());
        }
    } else {
        require(tsg::configure_cuda_launch_policy(), "cannot configure the CUDA launch policy");
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
    e->weights_ctx = ggml_init({size_t(gguf_get_n_tensors(file) + 1) * ggml_tensor_overhead() + 4096, nullptr, true});
    require(e->weights_ctx != nullptr, "cannot allocate weight metadata");
    std::map<std::string, std::vector<std::string>> fused;
    std::unordered_set<std::string> replaced;
    const auto * precise_setting = std::getenv("TSG_EMBEDDING_F32_LINEAR");
    bool validated_cpu_geometry = is_gemma2 && ggml_backend_is_cpu(e->backend) &&
        e->dim == 512 && e->output_dim == 768 && e->heads == 4 && e->layers == 24 && e->ff == 2048 &&
        e->context == 8192 && e->per_layer_dim == 512 && e->full_head == 512 && e->swa_head == 256 &&
        e->full_rot == 512 && e->swa_rot == 256 && e->swa_window == 1024;
    if (validated_cpu_geometry) for (int l = 0; l < e->layers; ++l)
        validated_cpu_geometry = validated_cpu_geometry && e->sliding[l] == (l % 6 < 5) &&
            e->kv_heads[l] == (e->sliding[l] ? 2 : 1);
#if (defined(__linux__) && defined(__x86_64__)) || (defined(__APPLE__) && defined(__aarch64__))
    // Qualified Linux and macOS ARM CPU Q/K/V shapes retain their exact F32 arithmetic while
    // assigning head*token rows, rather than only heads, across the worker team.
    // The helper retains upstream RMS below its measured token crossover.
    // Default only the measured seven/eight-worker configurations; other worker
    // counts retain the ordinary path until their own timing qualification.
    e->cpu_flat_qkv_rms = validated_cpu_geometry && ((threads > 0 ? threads : 4) == 7 || (threads > 0 ? threads : 4) == 8);
#endif
#if defined(TSG_EMBEDDING_CPU_TILED_GEMM)
    e->cpu_tiled_linears = validated_cpu_geometry && ggml_cpu_has_avx();
#endif
    // CPU Q8 matrix products also quantize input activations. The released
    // model's raw media rows expose that accumulated error in the complete HF
    // embedding oracle. Preserve the original weights as F32 for this qualified
    // geometry; token rows retain their original GGUF storage. A process-start
    // value of 0 restores the old arithmetic for explicit diagnostics.
    const bool precise_linear = (precise_setting && std::strcmp(precise_setting, "1") == 0) ||
        (validated_cpu_geometry && !(precise_setting && std::strcmp(precise_setting, "0") == 0));
    const bool cuda_linears = is_gemma2 && name == "CUDA";
    auto linear_storage = [&](const std::string & name, const ggml_tensor * source) {
        if (cuda_linears && source->ne[1] > 1 && name != "token_embd.weight") {
            // CUDA's quantized GEMV/GEMM paths quantize activations differently
            // across batch shapes. Dense half weights retain tensor-core GEMM
            // without that discontinuity. Raw hidden/media projections stay in
            // F32 because those inputs are not constrained by a preceding norm.
            if (name == "output.weight" || name == "per_layer_model_proj.weight" ||
                name.find(".inp_gate.weight") != std::string::npos) return GGML_TYPE_F32;
            return GGML_TYPE_F16;
        }
        // Apple's CPU has no native BF16 dot product. Keeping these matrices in
        // F32 also avoids truncating activations back to BF16 every layer. The
        // large vocabulary matrix stays in its original GGUF representation.
        // The tiny output projection must use F32 activations: quantizing after
        // pooling otherwise changes the per-token projection's embedding.
        if (is_gemma2 && ggml_backend_is_cpu(e->backend) && source->ne[1] > 1 && name != "token_embd.weight" &&
            (source->type == GGML_TYPE_BF16 || name == "output.weight" ||
             (precise_linear && ggml_is_quantized(source->type)))) return GGML_TYPE_F32;
        return source->type;
    };
    for (int layer = 0; layer < e->layers; ++layer) {
        const auto p = "blk." + std::to_string(layer) + ".";
        auto * q = ggml_get_tensor(source.tensors, (p + "attn_q.weight").c_str());
        auto * k = ggml_get_tensor(source.tensors, (p + "attn_k.weight").c_str());
        auto * v = ggml_get_tensor(source.tensors, (p + "attn_v.weight").c_str());
        auto * qb = ggml_get_tensor(source.tensors, (p + "attn_q.bias").c_str());
        auto * kb = ggml_get_tensor(source.tensors, (p + "attn_k.bias").c_str());
        auto * vb = ggml_get_tensor(source.tensors, (p + "attn_v.bias").c_str());
        const bool matching_bias = (!qb && !kb && !vb) || (qb && kb && vb && qb->type == kb->type && qb->type == vb->type &&
            q && k && v && ggml_nelements(qb) == q->ne[1] && ggml_nelements(kb) == k->ne[1] && ggml_nelements(vb) == v->ne[1]);
        if (!q || !k || !v || q->type != k->type || q->type != v->type || !matching_bias ||
            q->ne[0] != e->dim || k->ne[0] != e->dim || v->ne[0] != e->dim ||
            q->ne[2] != 1 || q->ne[3] != 1 || k->ne[2] != 1 || k->ne[3] != 1 || v->ne[2] != 1 || v->ne[3] != 1 ||
            (!is_gemma2 && (!ggml_are_same_shape(q, k) || !ggml_are_same_shape(q, v)))) continue;
        // Concatenate rows once while loading, preserving the original GGUF quantization.
        // One QKV projection reduces kernel launches and reuses the input activation tile.
        for (const auto * suffix : {"weight", "bias"}) {
            if (std::strcmp(suffix, "bias") == 0 && !qb) continue;
            const auto name = p + "attn_qkv." + suffix;
            const int64_t rows = q->ne[1] + k->ne[1] + v->ne[1];
            auto * tensor = std::strcmp(suffix, "weight") == 0 ? ggml_new_tensor_2d(e->weights_ctx, linear_storage(name, q), e->dim, rows) :
                ggml_new_tensor_1d(e->weights_ctx, qb->type, rows);
            ggml_set_name(tensor, name.c_str()); e->weights.emplace(name, tensor);
            auto & sources = fused[name];
            for (const auto * component : {"attn_q.", "attn_k.", "attn_v."}) {
                auto source_name = p + component + suffix;
                sources.push_back(source_name); replaced.insert(source_name);
            }
        }
    }
    if (is_gemma2) for (int layer = 0; layer < e->layers; ++layer) {
        const auto p = "blk." + std::to_string(layer) + ".";
        auto * gate = ggml_get_tensor(source.tensors, (p + "ffn_gate.weight").c_str());
        auto * up = ggml_get_tensor(source.tensors, (p + "ffn_up.weight").c_str());
        if (!gate || !up || gate->type != up->type || !ggml_are_same_shape(gate, up) ||
            gate->ne[0] != e->dim || gate->ne[1] != e->ff || gate->ne[2] != 1 || gate->ne[3] != 1) continue;
        const auto name = p + "ffn_gate_up.weight";
        auto * tensor = ggml_new_tensor_2d(e->weights_ctx, linear_storage(name, gate), e->dim, 2 * e->ff);
        ggml_set_name(tensor, name.c_str()); e->weights.emplace(name, tensor);
        for (const auto * component : {"ffn_gate.weight", "ffn_up.weight"}) {
            const auto source_name = p + component;
            fused[name].push_back(source_name); replaced.insert(source_name);
        }
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(file); ++i) {
        const std::string tensor_name = gguf_get_tensor_name(file, i);
        // Classification heads are not part of the sentence embedding graph.
        if (tensor_name.rfind("cls", 0) == 0) continue;
        if (replaced.count(tensor_name)) continue;
        auto * original = ggml_get_tensor(source.tensors, tensor_name.c_str());
        const auto storage = linear_storage(tensor_name, original);
        auto * tensor = storage == original->type ? ggml_dup_tensor(e->weights_ctx, original) :
            ggml_new_tensor(e->weights_ctx, storage, GGML_MAX_DIMS, original->ne);
        ggml_set_name(tensor, tensor_name.c_str()); e->weights.emplace(tensor_name, tensor);
    }
    e->validate();
    e->cpu_float_linears = std::all_of(e->weights.begin(), e->weights.end(), [](const auto & item) {
        const auto * value = item.second;
        return item.first == "token_embd.weight" || value->ne[1] == 1 || value->type == GGML_TYPE_F32;
    });
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
            require(original && destination_offset <= ggml_nbytes(tensor), "source exceeds tensor storage: " + part);
            input.seekg(gguf_get_data_offset(file) + gguf_get_tensor_offset(file, i));
            if (tensor->type != original->type && (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16)) {
                const auto * traits = ggml_get_type_traits(original->type);
                require(traits && (traits->to_float || original->type == GGML_TYPE_F32) && repack_data.empty(),
                    "cannot convert embedding linear weights: " + part);
                std::vector<char> data(ggml_nbytes(original));
                require(bool(input.read(data.data(), data.size())), "truncated tensor " + part);
                std::vector<float> converted(size_t(ggml_nelements(original)));
                const size_t element_bytes = tensor->type == GGML_TYPE_F32 ? sizeof(float) : sizeof(ggml_fp16_t);
                require(converted.size() * element_bytes <= ggml_nbytes(tensor) - destination_offset,
                    "converted source exceeds tensor storage: " + part);
                if (original->type == GGML_TYPE_F32) std::memcpy(converted.data(), data.data(), data.size());
                else traits->to_float(data.data(), converted.data(), converted.size());
                if (tensor->type == GGML_TYPE_F32) {
                    ggml_backend_tensor_set(tensor, converted.data(), destination_offset, converted.size() * sizeof(float));
                } else {
                    std::vector<ggml_fp16_t> half(converted.size());
                    ggml_fp32_to_fp16_row(converted.data(), half.data(), half.size());
                    ggml_backend_tensor_set(tensor, half.data(), destination_offset, half.size() * sizeof(ggml_fp16_t));
                }
                destination_offset += converted.size() * element_bytes;
                continue;
            }
            require(ggml_nbytes(original) <= ggml_nbytes(tensor) - destination_offset,
                "source exceeds tensor storage: " + part);
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
TSG_EXPORT int TSGgml_EmbeddingEncodeInputs(void * handle, const float * inputs, int input_capacity,
                                          const int32_t * lengths, int batch, float * output, int output_capacity) {
    try {
        tsg_embeddings::require(handle != nullptr, "null encoder");
        static_cast<tsg_embeddings::encoder *>(handle)->encode(nullptr, lengths, batch, output, output_capacity, inputs, input_capacity);
        return 0;
    } catch (const std::exception & e) { tsg::set_last_error(e.what()); return -1; }
}
TSG_EXPORT int TSGgml_EmbeddingTokenRows(void * handle, const int32_t * tokens, int count, float * output, int capacity) {
    try {
        tsg_embeddings::require(handle != nullptr, "null encoder");
        static_cast<tsg_embeddings::encoder *>(handle)->token_rows(tokens, count, output, capacity);
        return 0;
    } catch (const std::exception & e) { tsg::set_last_error(e.what()); return -1; }
}
