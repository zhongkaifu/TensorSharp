// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Qwen-Image-2.1: single-stream, segmented causal/image attention and shared AdaLN.
// All operations use unchanged upstream ggml. One complete graph per velocity
// prediction keeps intermediate activations on the device and weights resident.
// CUDA and Metal retain graph metadata and allocations across denoising steps;
// upstream ggml-cuda captures a retained graph as a CUDA graph once two
// consecutive calls leave it unchanged, so every retained step replays one.
//
// Prefix KV cache. Text and reference-image tokens are modulated with the t=0
// row and never attend to the target block, so their hidden states, and hence
// every layer's post-RoPE K and V, are identical at every denoising step
// (Qwen-Image-2.1 README "Prefix KV Cache"; diffusers QwenImage21KVCache). The
// first forward of a request runs the whole sequence and copies each layer's
// prefix K/V into a device buffer ("extract"); later forwards run only the
// target tokens and attend over [cached prefix; target] ("cached"). By default
// the cache holds exactly what the attention kernel reads (F16 for Metal and
// CUDA flash attention, F32 elsewhere), so cached steps reproduce the uncached
// computation. Optional Q8_0 storage is the 8-bit counterpart of vLLM-Omni's
// FP8 prefix cache: it is dequantized before attention, so only the prefix is
// rounded. A cache that does not fit the device is declined with a warning and
// the request continues on the whole-sequence graph.
#include "ggml_ops_internal.h"
#include "ggml_ops_qwen_image21.h"

using namespace tsg;
namespace {
struct Upload { ggml_tensor* tensor; const void* data; size_t bytes; };
struct Resident { const void* key; size_t bytes; ggml_backend_buffer_t buffer; };
// Per-step inputs: which descriptor array feeds a graph input, and from where.
// kHead is the adapter's per-call output head (F16 or F32); the others are F32.
enum Field { kImages, kText, kTime, kCos, kSin, kHead };
struct InputBinding { ggml_tensor* tensor; Field field; size_t offset; };
enum class Mode { Full, Extract, Cached };

bool option_enabled(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    return value ? value[0] != '0' : default_value;
}

// Start of an input's data; `offset` counts F32 elements for the F32 fields.
const void* field_data(const TSGQi21Desc& d, Field field, size_t offset) {
    switch (field) {
        case kImages: return d.images + offset;
        case kText: return d.text + offset;
        case kTime: return d.time_embedding + offset;
        case kCos: return d.cos + offset;
        case kSin: return d.sin + offset;
        case kHead: return d.adapter ? d.adapter->output_head : nullptr;
    }
    return nullptr;
}

bool has_update(const TSGQi21Lora& l) { return l.rank > 0 || l.row_scale; }

size_t lora_down_bytes(const TSGQi21Lora& l) {
    return ggml_row_size(static_cast<ggml_type>(l.type), l.in) * static_cast<size_t>(l.rank);
}

size_t lora_up_bytes(const TSGQi21Lora& l) {
    return ggml_row_size(static_cast<ggml_type>(l.type), l.rank) * static_cast<size_t>(l.out);
}

// Attention reads F16 K/V with flash attention on Metal, and on CUDA unless the
// padded-mask diagnostic is selected; every other configuration reads F32.
ggml_type attention_kv_type() {
    const bool flash = option_enabled("TS_QWEN21_FLASH", true);
    if (flash && (g_backend_type == BACKEND_TYPE_METAL ||
        (g_backend_type == BACKEND_TYPE_CUDA && !option_enabled("TS_QWEN21_PAD_MASK", false))))
        return GGML_TYPE_F16;
    return GGML_TYPE_F32;
}

// Stored K/V of one request's prefix, on the device that computed it.
struct PrefixCache {
    std::uint64_t key = 0, id = 0;
    int rank = 0;
    ggml_backend_t backend = nullptr;
    ggml_type k_type = GGML_TYPE_F16, v_type = GGML_TYPE_F16, attn_type = GGML_TYPE_F16;
    std::int32_t requested_type = TSG_QI21_PREFIX_AUTO;
    bool filled = false, declined = false;
    size_t bytes = 0;
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_tensor*> k, v;
    TSGQi21Desc shape{};
    std::vector<TSGQi21Segment> segments;
    std::vector<TSGQi21Weight> weight_key;
    bool flash = true, pad_mask = false;
    uint64_t used = 0;

    PrefixCache() = default;
    PrefixCache(const PrefixCache&) = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;
    ~PrefixCache() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
};

bool same_weight(const TSGQi21Weight& a, const TSGQi21Weight& b) {
    return a.data == b.data && a.type == b.type && a.ne0 == b.ne0 &&
        a.ne1 == b.ne1 && a.bytes == b.bytes;
}

bool same_segments(const TSGQi21Segment* a, const TSGQi21Segment* b, int count) {
    for (int i = 0; i < count; ++i)
        if (a[i].start != b[i].start || a[i].end != b[i].end ||
            a[i].source_start != b[i].source_start || a[i].is_image != b[i].is_image) return false;
    return true;
}

// A LoRA's buffers as weight-key entries: a retained graph or stored prefix built with
// other factors (or none) must never serve this adapter. The shared-shrink layout is
// derived from these pointers, so it is covered too.
void append_lora(std::vector<TSGQi21Weight>& key, const TSGQi21Lora& l) {
    key.push_back({l.down, l.type, l.rank, l.in, l.out, static_cast<int64_t>(l.rank > 0 ? lora_down_bytes(l) : 0)});
    key.push_back({l.up, l.type, l.rank, l.rank, l.out, static_cast<int64_t>(l.rank > 0 ? lora_up_bytes(l) : 0)});
    key.push_back({l.row_scale, GGML_TYPE_F32, 0, l.out, 1, l.row_scale ? l.out * int64_t(sizeof(float)) : 0});
}

std::vector<TSGQi21Weight> weights(const TSGQi21Desc& d) {
    std::vector<TSGQi21Weight> result = {d.image_in, d.text_in, d.text_out,
        d.time_in, d.time_out, d.modulation, d.norm_out, d.proj_out};
    result.push_back({d.text_norm, GGML_TYPE_F32, 0, d.text_dim, 1, d.text_dim * int64_t(sizeof(float))});
    for (int i = 0; i < d.num_layers; ++i) {
        const auto& b = d.blocks[i];
        for (const auto* w : {&b.q, &b.k, &b.v, &b.out, &b.gate, &b.up, &b.down}) result.push_back(*w);
        for (void* p : {b.norm_q, b.norm_k})
            result.push_back({p, GGML_TYPE_F32, 0, d.head_dim, 1, d.head_dim * int64_t(sizeof(float))});
    }
    if (const TSGQi21Adapter* a = d.adapter) {
        for (const auto* l : {&a->image_in, &a->text_in, &a->text_out, &a->time_in, &a->time_out,
                              &a->modulation, &a->norm_out, &a->proj_out}) append_lora(result, *l);
        if (a->blocks)
            for (int i = 0; i < d.num_layers; ++i) {
                const auto& l = a->blocks[i];
                for (const auto* u : {&l.q, &l.k, &l.v, &l.out, &l.gate, &l.up, &l.down}) append_lora(result, *u);
            }
        // The head's data is a per-call input; only its presence and type shape the graph.
        if (a->output_head)
            result.push_back({nullptr, a->output_head_type, 1, d.dim, d.channels, 0});
    }
    return result;
}

bool same_weights(const std::vector<TSGQi21Weight>& a, const std::vector<TSGQi21Weight>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (!same_weight(a[i], b[i])) return false;
    return true;
}

struct Builder {
    ggml_context* ctx;
    std::vector<Upload> constants;
    std::vector<InputBinding> inputs;
    std::vector<std::vector<ggml_fp16_t>> masks;
    std::vector<Resident> resident;

    void bind(ggml_tensor* t, const void* data, size_t bytes) {
        if (!data || bytes != ggml_nbytes(t)) throw std::invalid_argument("QwenImage21: invalid weight descriptor");
        ggml_backend_buffer_t buffer = nullptr;
        void* address = nullptr;
        bool needs_upload = false;
        void* key = const_cast<void*>(data);
        if (try_get_cacheable_tensor_buffer(g_backend, ggml_backend_get_device(g_backend), t,
                key, bytes, buffer, address, needs_upload)) {
            if (ggml_backend_tensor_alloc(buffer, t, address) == GGML_STATUS_SUCCESS) {
                // A failed graph allocation must not leave an uninitialized resident weight.
                if (needs_upload) { host_read_barrier(); ggml_backend_tensor_set(t, data, 0, bytes); }
                resident.push_back({data, bytes, t->buffer});
                return;
            }
            invalidate_cached_buffer(key);
        }
        // A residency-budget refusal does not invalidate another CFG graph's
        // private weight copy. Each graph can retain its own constant input.
        ggml_set_input(t);
        constants.push_back({t, data, bytes});
    }
    ggml_tensor* weight(const TSGQi21Weight& w) {
        if (w.ne0 <= 0 || w.ne1 <= 0 || w.type < 0 || w.type >= GGML_TYPE_COUNT)
            throw std::invalid_argument("QwenImage21: invalid weight shape/type");
        auto t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(w.type), w.ne0, w.ne1);
        bind(t, w.data, static_cast<size_t>(w.bytes));
        return t;
    }
    ggml_tensor* gain(void* p, int n) {
        auto t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        bind(t, p, n * sizeof(float));
        return t;
    }
    ggml_tensor* input(Field field, size_t offset, int n0, int n1, ggml_type type = GGML_TYPE_F32) {
        auto t = ggml_new_tensor_2d(ctx, type, n0, n1);
        ggml_set_input(t);
        inputs.push_back({t, field, offset});
        return t;
    }
    // x * factor with F32 accumulation: CUDA would otherwise run an F16 factor's
    // product in F16 and Metal keeps F32 accumulators anyway.
    ggml_tensor* lora_product(ggml_tensor* factor, ggml_tensor* x) {
        auto result = ggml_mul_mat(ctx, factor, x);
        ggml_prec_set_acc(result, GGML_PREC_F32);
        return result;
    }
    // The down projections of `updates` applied to one shared input: [rank, seq] per
    // update, null where an update has no low-rank term. Updates whose `down` factors sit
    // back to back in one allocation run as one stacked shrink (x is read once), and an
    // update that reuses the previous one's factor (a fused gate_up LoRA seen as its two
    // halves) reuses its product.
    std::vector<ggml_tensor*> shrink(std::initializer_list<const TSGQi21Lora*> updates, ggml_tensor* x) {
        std::vector<const TSGQi21Lora*> list(updates);
        std::vector<ggml_tensor*> result(list.size(), nullptr);
        for (size_t i = 0; i < list.size();) {
            const TSGQi21Lora* first = list[i];
            if (!first || first->rank <= 0) { ++i; continue; }
            if (x->ne[0] != first->in) throw std::invalid_argument("QwenImage21: LoRA input shape mismatch");
            // Extend the run while the next factor starts where this one ends.
            size_t end = i + 1;
            int64_t rank = first->rank;
            const char* next = static_cast<const char*>(first->down) + lora_down_bytes(*first);
            while (end < list.size() && list[end] && list[end]->rank > 0 && list[end]->type == first->type &&
                   list[end]->in == first->in && list[end]->down == next) {
                rank += list[end]->rank;
                next += lora_down_bytes(*list[end]);
                ++end;
            }
            auto down = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(first->type), first->in, rank);
            bind(down, first->down, static_cast<size_t>(next - static_cast<const char*>(first->down)));
            auto t = lora_product(down, x);
            size_t offset = 0;
            for (size_t j = i; j < end; ++j) {
                result[j] = end - i == 1 ? t :
                    ggml_view_2d(ctx, t, list[j]->rank, t->ne[1], t->nb[1], offset * sizeof(float));
                offset += static_cast<size_t>(list[j]->rank);
            }
            // Shared factor: the same bytes describe the next updates' down projections.
            if (end - i == 1)
                while (end < list.size() && list[end] && list[end]->rank > 0 &&
                       list[end]->down == first->down && list[end]->rank == first->rank &&
                       list[end]->type == first->type && list[end]->in == first->in)
                    result[end++] = t;
            i = end;
        }
        return result;
    }
    // base (x's projection) plus an update: the DoRA row scale on the base, then the
    // low-rank term up * shrunk, where shrunk = down * x from shrink().
    ggml_tensor* adapt(ggml_tensor* base, const TSGQi21Lora& l, ggml_tensor* shrunk) {
        if (l.out != base->ne[0]) throw std::invalid_argument("QwenImage21: LoRA output shape mismatch");
        if (l.row_scale) base = ggml_mul(ctx, base, gain(l.row_scale, static_cast<int>(l.out)));
        if (l.rank <= 0) return base;
        auto up = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(l.type), l.rank, l.out);
        bind(up, l.up, lora_up_bytes(l));
        return ggml_add(ctx, base, lora_product(up, shrunk));
    }
    ggml_tensor* linear(const TSGQi21Weight& w, ggml_tensor* x, const TSGQi21Lora* lora = nullptr,
                        ggml_tensor* shrunk = nullptr) {
        if (x->ne[0] != w.ne0) throw std::invalid_argument("QwenImage21: projection shape mismatch");
        auto y = ggml_mul_mat(ctx, weight(w), x);
        if (!lora || !has_update(*lora)) return y;
        if (lora->rank > 0 && !shrunk) shrunk = shrink({lora}, x)[0];
        return adapt(y, *lora, shrunk);
    }
    ggml_tensor* slice(ggml_tensor* x, int start, int length) {
        return ggml_view_2d(ctx, x, x->ne[0], length, x->nb[1], start * x->nb[1]);
    }
    ggml_tensor* modulate(ggml_tensor* x, ggml_tensor* m, int prefix, bool gate) {
        auto target = slice(m, 0, 1);
        auto preceding = slice(m, 1, 1);
        target = gate ? ggml_tanh(ctx, target) : ggml_scale_bias(ctx, target, 1.f, 1.f);
        preceding = gate ? ggml_tanh(ctx, preceding) : ggml_scale_bias(ctx, preceding, 1.f, 1.f);
        auto result = ggml_mul(ctx, slice(x, prefix, x->ne[1] - prefix), target);
        if (prefix) result = ggml_concat(ctx, ggml_mul(ctx, slice(x, 0, prefix), preceding), result, 1);
        return result;
    }
    ggml_tensor* rope(ggml_tensor* x, ggml_tensor* cos, ggml_tensor* sin, int hd, int heads, int seq) {
        const int half = hd / 2;
        auto x4 = ggml_reshape_4d(ctx, x, 2, half, heads, seq);
        auto e = ggml_view_4d(ctx, x4, 1, half, heads, seq, x4->nb[1], x4->nb[2], x4->nb[3], 0);
        auto o = ggml_view_4d(ctx, x4, 1, half, heads, seq, x4->nb[1], x4->nb[2], x4->nb[3], sizeof(float));
        auto c = ggml_reshape_4d(ctx, cos, 1, half, 1, seq);
        auto s = ggml_reshape_4d(ctx, sin, 1, half, 1, seq);
        auto ep = ggml_reshape_3d(ctx, ggml_sub(ctx, ggml_mul(ctx, e, c), ggml_mul(ctx, o, s)), half, heads, seq);
        auto op = ggml_reshape_3d(ctx, ggml_add(ctx, ggml_mul(ctx, o, c), ggml_mul(ctx, e, s)), half, heads, seq);
        // Q and K share the same half-split permutation; their dot product is
        // unchanged, and this avoids a slow four-dimensional interleave concat.
        return ggml_concat(ctx, ep, op, 0);
    }
    // Conversion of stored cache values to the attention's K/V type. CUDA has
    // no direct Q8_0-to-F16 copy, so it dequantizes through F32.
    ggml_tensor* convert(ggml_tensor* x, ggml_type type) {
        if (x->type == type) return x;
        auto direct = ggml_cast(ctx, x, type);
        if (x->type == GGML_TYPE_F32 || type == GGML_TYPE_F32 || backend_supports_op(direct)) return direct;
        return ggml_cast(ctx, ggml_cast(ctx, x, GGML_TYPE_F32), type);
    }
    // Keys or values of the whole sequence for a cached step: the stored prefix
    // followed by this step's target rows, in the layout attention reads.
    ggml_tensor* joined(ggml_tensor* stored, ggml_tensor* current, ggml_type type) {
        auto tail = ggml_permute(ctx, current, 0, 2, 1, 3);
        tail = type == GGML_TYPE_F32 ? ggml_cont(ctx, tail) : ggml_cast(ctx, tail, type);
        return ggml_concat(ctx, convert(stored, type), tail, 1);
    }
    // qp holds query rows [q_base, q_base + rows) as [head_dim, rows, heads].
    // kp/vp hold key rows [0, nk) and may already be F16 (a cached step); they
    // are then consumed without a second conversion.
    ggml_tensor* attention(ggml_tensor* qp, ggml_tensor* kp, ggml_tensor* vp, int q_base,
                          const std::vector<TSGQi21Segment>& segments, int heads, int head_dim,
                          const std::vector<ggml_tensor*>& mask_tensors) {
        const bool flash = option_enabled("TS_QWEN21_FLASH", true);
        const bool shared_half = flash && g_backend_type == BACKEND_TYPE_CUDA &&
            !option_enabled("TS_QWEN21_PAD_MASK", false);
        // CUDA flash attention converts F32 K/V to F16 internally. Convert once
        // here so every segment can share them, including strided prefix views.
        const bool converted = kp->type != GGML_TYPE_F32;
        auto kflash = shared_half && !converted ? ggml_cast(ctx, kp, GGML_TYPE_F16) : kp;
        auto vflash = shared_half && !converted ? ggml_cast(ctx, vp, GGML_TYPE_F16) : vp;
        const int width = heads * head_dim;
        ggml_tensor* joined_out = nullptr;
        const float scale = 1.f / std::sqrt(static_cast<float>(head_dim));
        for (size_t i = 0; i < segments.size(); ++i) {
            const auto& seg = segments[i];
            int nq = seg.end - seg.start, nk = seg.end;
            auto qs = ggml_view_3d(ctx, qp, head_dim, nq, heads, qp->nb[1], qp->nb[2], (seg.start - q_base) * qp->nb[1]);
            auto ks = ggml_view_3d(ctx, kp, head_dim, nk, heads, kp->nb[1], kp->nb[2], 0);
            auto vs = ggml_view_3d(ctx, vp, head_dim, nk, heads, vp->nb[1], vp->nb[2], 0);
            ggml_tensor* out = nullptr;
            if (flash) {
                // Bidirectional image segments need no mask when the backend
                // accepts their actual KV length. Let upstream handle its own
                // tile boundaries instead of materializing a quadratic padding
                // mask and padded F32 K/V for every layer.
                int pad = !shared_half && mask_tensors[i] ? mask_tensors[i]->ne[0] - nk : 0;
                auto kpart = shared_half ? ggml_view_3d(ctx, kflash, head_dim, nk, heads, kflash->nb[1], kflash->nb[2], 0) : ks;
                auto vpart = shared_half ? ggml_view_3d(ctx, vflash, head_dim, nk, heads, vflash->nb[1], vflash->nb[2], 0) : vs;
                auto kpad = pad ? ggml_pad(ctx, kpart, 0, pad, 0, 0) : kpart;
                auto vpad = pad ? ggml_pad(ctx, vpart, 0, pad, 0, 0) : vpart;
                if (g_backend_type == BACKEND_TYPE_METAL) {
                    // CAST accepts the strided segment directly and produces
                    // contiguous F16, avoiding an intermediate F32 CONT copy.
                    if (kpad->type == GGML_TYPE_F32) kpad = ggml_cast(ctx, kpad, GGML_TYPE_F16);
                    if (vpad->type == GGML_TYPE_F32) vpad = ggml_cast(ctx, vpad, GGML_TYPE_F16);
                } else if (!shared_half) {
                    if (!ggml_is_contiguous(kpad)) kpad = ggml_cont(ctx, kpad);
                    if (!ggml_is_contiguous(vpad)) vpad = ggml_cont(ctx, vpad);
                }
                auto fa = ggml_flash_attn_ext(ctx, qs, kpad, vpad, mask_tensors[i], scale, 0.f, 0.f);
                ggml_prec_set_acc(fa, GGML_PREC_F32);
                if (backend_supports_op(fa)) out = ggml_reshape_2d(ctx, fa, width, nq);
            }
            if (!out) {
                auto scores = ggml_mul_mat(ctx, ks, qs);
                ggml_prec_set_acc(scores, GGML_PREC_F32);
                ggml_tensor* causal = nullptr;
                if (!seg.is_image) {
                    // DIAG_MASK_INF is unavailable in unchanged upstream Metal.
                    // Supply the exact same causal mask to the softmax instead.
                    causal = ggml_view_2d(ctx, mask_tensors[i], nk, nq, mask_tensors[i]->nb[1], 0);
                    causal = ggml_cast(ctx, ggml_cont(ctx, causal), GGML_TYPE_F32);
                }
                auto probs = ggml_soft_max_ext(ctx, scores, causal, scale, 0.f);
                auto vt = ggml_cont(ctx, ggml_permute(ctx, vs, 1, 0, 2, 3));
                auto av = ggml_mul_mat(ctx, vt, probs);
                out = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, av, 0, 2, 1, 3)), width, nq);
            }
            joined_out = joined_out ? ggml_concat(ctx, joined_out, out, 1) : out;
        }
        return joined_out;
    }
};

struct ForwardGraph {
    ggml_context* ctx = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph* graph = nullptr;
    std::vector<InputBinding> inputs;
    ggml_tensor* output = nullptr;
    ggml_backend_t backend = nullptr;
    TSGQi21Desc shape{};
    std::vector<TSGQi21Weight> weight_key;
    std::vector<TSGQi21Segment> segments;
    std::vector<Resident> resident;
    bool flash = true, pad_mask = false;
    Mode mode = Mode::Full;
    uint64_t cache_id = 0;
    uint64_t used = 0, runs = 0;
    // Tensor parallelism: the row-parallel partial sums the ranks all-reduce,
    // and the segment schedule that stops after each of them.
    std::vector<ggml_tensor*> boundaries;
    TpRankPlan plan;

    ~ForwardGraph() {
        if (allocator) ggml_gallocr_free(allocator);
        if (ctx) ggml_free(ctx);
    }

    bool matches(const TSGQi21Desc& d, const std::vector<TSGQi21Weight>& key, Mode wanted, uint64_t wanted_cache) const {
        if (backend != g_backend || mode != wanted || cache_id != wanted_cache ||
            shape.dim != d.dim || shape.heads != d.heads ||
            shape.head_dim != d.head_dim || shape.channels != d.channels || shape.text_dim != d.text_dim ||
            shape.image_seq != d.image_seq || shape.text_seq != d.text_seq || shape.total_seq != d.total_seq ||
            shape.prefix_seq != d.prefix_seq || shape.num_layers != d.num_layers || shape.num_segments != d.num_segments ||
            shape.tp_ranks != d.tp_ranks ||
            shape.eps != d.eps || flash != option_enabled("TS_QWEN21_FLASH", true) ||
            pad_mask != option_enabled("TS_QWEN21_PAD_MASK", false) || !same_weights(key, weight_key) ||
            !same_segments(segments.data(), d.segments, d.num_segments)) return false;
        // Weight cache entries can be replaced outside this path. Inspect the
        // live maps without dereferencing a possibly freed backend buffer.
        std::scoped_lock lock(g_host_buffer_cache_mutex, g_preloaded_buffer_cache_mutex);
        for (const auto& r : resident) {
            auto found = [&](const auto& cache) {
                auto it = cache.find(const_cast<void*>(r.key));
                return it != cache.end() && it->second.buffer == r.buffer && it->second.bytes == r.bytes;
            };
            if (!found(g_host_buffer_cache) && !found(g_preloaded_buffer_cache)) return false;
        }
        return true;
    }
};

std::recursive_mutex graph_mutex;
std::array<std::array<std::unique_ptr<ForwardGraph>, 2>, TSG_MAX_DEVICES> graph_cache;
uint64_t graph_use = 0;
// Requests normally hold one cache per CFG branch; the bound only protects a
// caller that never releases its keys.
constexpr size_t kMaxPrefixCaches = 4;
std::vector<std::unique_ptr<PrefixCache>> prefix_caches;
uint64_t prefix_cache_ids = 0, prefix_cache_use = 0;

void validate(const TSGQi21Desc* d) {
        if (!d || d->struct_bytes != sizeof(TSGQi21Desc) || !d->images || !d->text || !d->output ||
            !d->time_embedding || !d->cos || !d->sin || !d->blocks || !d->segments ||
            d->dim <= 0 || d->head_dim <= 0 || d->head_dim % 2 || d->heads <= 0 ||
            d->tp_ranks < 0 || d->tp_ranks > TSG_MAX_DEVICES ||
            d->dim != d->heads * d->head_dim * std::max(1, d->tp_ranks) || d->channels <= 0 || d->text_dim <= 0 ||
            d->num_layers <= 0 || d->num_segments <= 0 || d->image_seq <= 0 || d->text_seq <= 0 ||
            d->prefix_seq < 0 || d->total_seq <= d->prefix_seq || !std::isfinite(d->eps) || d->eps <= 0 ||
            d->prefix_cache_type < TSG_QI21_PREFIX_AUTO || d->prefix_cache_type > TSG_QI21_PREFIX_Q8_0_V)
            throw std::invalid_argument("QwenImage21: invalid forward descriptor");
        int end = 0;
        for (int i = 0; i < d->num_segments; ++i) {
            const auto& s = d->segments[i];
            if (s.start != end || s.end <= s.start || s.end > d->total_seq || s.source_start < 0 ||
                s.source_start + s.end - s.start > (s.is_image ? d->image_seq : d->text_seq))
                throw std::invalid_argument("QwenImage21: invalid sequence segment");
            end = s.end;
        }
        if (end != d->total_seq || d->segments[d->num_segments - 1].start != d->prefix_seq ||
            !d->segments[d->num_segments - 1].is_image)
            throw std::invalid_argument("QwenImage21: missing target image segment");
        if (const TSGQi21Adapter* a = d->adapter) {
            if (a->struct_bytes != sizeof(TSGQi21Adapter) || a->num_layers != d->num_layers)
                throw std::invalid_argument("QwenImage21: invalid LoRA adapter descriptor");
            // Every update must fit the projection it modifies; the graph also checks the
            // shapes it actually builds, but a bad descriptor should fail before any work.
            auto check = [](const TSGQi21Lora& l, const TSGQi21Weight& w, const char* what) {
                if (!has_update(l)) return;
                const bool typed = l.type == GGML_TYPE_F16 || l.type == GGML_TYPE_F32;
                if (l.rank < 0 || l.in != w.ne0 || l.out != w.ne1 || !w.data ||
                    (l.rank > 0 && (!typed || !l.down || !l.up)))
                    throw std::invalid_argument(std::string("QwenImage21: LoRA update does not fit ") + what);
            };
            check(a->image_in, d->image_in, "img_in");
            check(a->text_in, d->text_in, "txt_in.in_layer");
            check(a->text_out, d->text_out, "txt_in.out_layer");
            check(a->time_in, d->time_in, "timestep_embedder.linear_1");
            check(a->time_out, d->time_out, "timestep_embedder.linear_2");
            check(a->modulation, d->modulation, "modulation.1");
            check(a->norm_out, d->norm_out, "norm_out.linear");
            check(a->proj_out, d->proj_out, "proj_out");
            if (a->output_head && ((a->output_head_type != GGML_TYPE_F16 && a->output_head_type != GGML_TYPE_F32) ||
                                   has_update(a->proj_out)))
                throw std::invalid_argument("QwenImage21: an output head replaces proj_out and cannot carry a LoRA");
            if (a->blocks)
                for (int i = 0; i < d->num_layers; ++i) {
                    const auto& l = a->blocks[i];
                    const auto& w = d->blocks[i];
                    check(l.q, w.q, "attn.to_q");
                    check(l.k, w.k, "attn.to_k");
                    check(l.v, w.v, "attn.to_v");
                    check(l.out, w.out, "attn.to_out.0");
                    check(l.gate, w.gate, "img_mlp gate");
                    if (w.up.data) check(l.up, w.up, "img_mlp up");
                    else if (has_update(l.up))
                        throw std::invalid_argument("QwenImage21: a fused gate_up weight takes its LoRA as one gate update");
                    check(l.down, w.down, "img_mlp.out");
                }
        }
}

// Retained graphs that read a cache must not outlive it.
void drop_graphs_of_cache(uint64_t id) {
    for (auto& device : graph_cache) for (auto& entry : device)
        if (entry && entry->cache_id == id) entry.reset();
}

void drop_prefix_cache(std::vector<std::unique_ptr<PrefixCache>>::iterator it) {
    drop_graphs_of_cache((*it)->id);
    prefix_caches.erase(it);
}

std::unique_ptr<ForwardGraph> build_graph(const TSGQi21Desc* d, bool persistent, Mode mode, PrefixCache* cache) {
        auto result = std::make_unique<ForwardGraph>();
        result->backend = g_backend;
        result->shape = *d;
        result->weight_key = weights(*d);
        result->segments.assign(d->segments, d->segments + d->num_segments);
        result->flash = option_enabled("TS_QWEN21_FLASH", true);
        result->pad_mask = option_enabled("TS_QWEN21_PAD_MASK", false);
        result->mode = mode;
        result->cache_id = cache ? cache->id : 0;
        const bool cached = mode == Mode::Cached;
        const TSGQi21Segment& target_segment = d->segments[d->num_segments - 1];
        // A cached step computes only the target rows; the prefix it attends to
        // was stored by the extract step.
        const std::vector<TSGQi21Segment> segments = cached ?
            std::vector<TSGQi21Segment>{target_segment} : result->segments;
        const int first = cached ? d->prefix_seq : 0;
        const int seq = d->total_seq - first;
        const int prefix = cached ? 0 : d->prefix_seq;
        const int num_segments = static_cast<int>(segments.size());
        // A LoRA adds at most a shrink, a row scale, an expand and an add per projection.
        size_t nodes = static_cast<size_t>(d->num_layers) * (190 + num_segments * 45 + (d->adapter ? 64 : 0)) + 1024 +
            (d->adapter ? 128 : 0);
        ggml_init_params init{ggml_tensor_overhead() * (nodes + 1024) + ggml_graph_overhead_custom(nodes, false), nullptr, true};
        result->ctx = ggml_init(init);
        if (!result->ctx) throw std::runtime_error("QwenImage21: graph context allocation failed");
        Builder b{result->ctx, {}, {}, {}, {}};
        auto ctx = result->ctx;
        auto graph = ggml_new_graph_custom(ctx, nodes, false);
        const int target_tokens = target_segment.end - target_segment.start;
        auto images = cached ?
            b.input(kImages, size_t(target_segment.source_start) * d->channels, d->channels, target_tokens) :
            b.input(kImages, 0, d->channels, d->image_seq);
        auto text = cached ? nullptr : b.input(kText, 0, d->text_dim, d->text_seq);
        // Both rows are computed even when only the target reads its row: one
        // two-row matmul keeps the target's modulation identical to the
        // whole-sequence graph's, whose kernel choice depends on the row count.
        auto time = b.input(kTime, 0, 256, 2);
        auto cos = b.input(kCos, size_t(first) * (d->head_dim / 2), d->head_dim / 2, seq);
        auto sin = b.input(kSin, size_t(first) * (d->head_dim / 2), d->head_dim / 2, seq);
        std::vector<ggml_tensor*> mask_tensors;
        b.masks.reserve(num_segments);
        for (const auto& s : segments) {
            // Bidirectional segments attend to exactly their prefix. CUDA's
            // tile/MMA kernels handle unpadded lengths without a dense mask.
            if (s.is_image && (g_backend_type == BACKEND_TYPE_CPU || g_backend_type == BACKEND_TYPE_METAL ||
                (g_backend_type == BACKEND_TYPE_CUDA && !result->pad_mask))) {
                mask_tensors.push_back(nullptr);
                continue;
            }
            int nk = (s.end + 255) / 256 * 256, nq = (s.end - s.start + 63) / 64 * 64;
            b.masks.emplace_back(static_cast<size_t>(nk) * nq, ggml_fp32_to_fp16(-INFINITY));
            auto& data = b.masks.back();
            for (int q = 0; q < s.end - s.start; ++q)
                for (int k = 0; k < (s.is_image ? s.end : s.start + q + 1); ++k) data[static_cast<size_t>(q) * nk + k] = 0;
            auto mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nk, nq);
            ggml_set_input(mask);
            b.constants.push_back({mask, data.data(), data.size() * sizeof(ggml_fp16_t)});
            mask_tensors.push_back(mask);
        }
        const TSGQi21Adapter* adapter = d->adapter;
        auto global = [adapter](TSGQi21Lora TSGQi21Adapter::*member) -> const TSGQi21Lora* {
            return adapter ? &(adapter->*member) : nullptr;
        };
        time = ggml_silu(ctx, b.linear(d->time_out,
            ggml_silu(ctx, b.linear(d->time_in, time, global(&TSGQi21Adapter::time_in))), global(&TSGQi21Adapter::time_out)));
        auto mod = b.linear(d->modulation, time, global(&TSGQi21Adapter::modulation));
        std::vector<ggml_tensor*> modulation;
        for (int i = 0; i < 4; ++i)
            modulation.push_back(ggml_cont(ctx, ggml_view_2d(ctx, mod, d->dim, 2, mod->nb[1], i * d->dim * sizeof(float))));
        images = b.linear(d->image_in, images, global(&TSGQi21Adapter::image_in));
        ggml_tensor* joint = nullptr;
        if (cached) {
            joint = images;
        } else {
            auto norm = ggml_scale_bias(ctx, b.gain(d->text_norm, d->text_dim), 1.f, 1.f);
            text = ggml_mul(ctx, ggml_rms_norm(ctx, text, d->eps), norm);
            text = b.linear(d->text_out, ggml_gelu(ctx, b.linear(d->text_in, text, global(&TSGQi21Adapter::text_in))),
                global(&TSGQi21Adapter::text_out));
            for (const auto& s : segments) {
                auto h = b.slice(s.is_image ? images : text, s.source_start, s.end - s.start);
                joint = joint ? ggml_concat(ctx, joint, h, 1) : h;
            }
        }
        const ggml_type attn_type = cache ? cache->attn_type : GGML_TYPE_F32;
        for (int i = 0; i < d->num_layers; ++i) {
            const auto& w = d->blocks[i];
            const TSGQi21BlockLora* l = adapter && adapter->blocks ? &adapter->blocks[i] : nullptr;
            auto h = b.modulate(ggml_norm(ctx, joint, d->eps), modulation[0], prefix, false);
            // Q, K and V read the same input: one stacked LoRA shrink serves all three.
            const auto qkv = l ? b.shrink({&l->q, &l->k, &l->v}, h) : std::vector<ggml_tensor*>(3, nullptr);
            auto q = ggml_reshape_3d(ctx, b.linear(w.q, h, l ? &l->q : nullptr, qkv[0]), d->head_dim, d->heads, seq);
            auto k = ggml_reshape_3d(ctx, b.linear(w.k, h, l ? &l->k : nullptr, qkv[1]), d->head_dim, d->heads, seq);
            auto v = ggml_reshape_3d(ctx, b.linear(w.v, h, l ? &l->v : nullptr, qkv[2]), d->head_dim, d->heads, seq);
            q = ggml_mul(ctx, ggml_rms_norm(ctx, q, d->eps), b.gain(w.norm_q, d->head_dim));
            k = ggml_mul(ctx, ggml_rms_norm(ctx, k, d->eps), b.gain(w.norm_k, d->head_dim));
            q = b.rope(q, cos, sin, d->head_dim, d->heads, seq);
            k = b.rope(k, cos, sin, d->head_dim, d->heads, seq);
            if (mode == Mode::Extract) {
                // Store this layer's prefix rows as the attention will read
                // them. Expanding here keeps K/V live only until this layer's
                // attention instead of the end of the graph.
                for (auto [source, stored] : {std::pair{k, cache->k[i]}, std::pair{v, cache->v[i]}}) {
                    auto rows = ggml_permute(ctx, source, 0, 2, 1, 3);
                    rows = ggml_view_3d(ctx, rows, d->head_dim, d->prefix_seq, d->heads, rows->nb[1], rows->nb[2], 0);
                    ggml_build_forward_expand(graph, ggml_cpy(ctx, rows, stored));
                }
            }
            auto qp = ggml_permute(ctx, q, 0, 2, 1, 3);
            ggml_tensor *kp, *vp;
            if (cached) {
                kp = b.joined(cache->k[i], k, attn_type);
                vp = b.joined(cache->v[i], v, attn_type);
            } else {
                kp = ggml_permute(ctx, k, 0, 2, 1, 3);
                vp = ggml_permute(ctx, v, 0, 2, 1, 3);
            }
            // A row-parallel rank adds its partial LoRA term before the all-reduce: the
            // sum over ranks of up * (down_r * x_r) is the whole update.
            h = b.linear(w.out, b.attention(qp, kp, vp, first, segments, d->heads, d->head_dim, mask_tensors),
                l ? &l->out : nullptr);
            if (d->tp_ranks > 1) result->boundaries.push_back(h);
            joint = ggml_add(ctx, joint, b.modulate(h, modulation[1], prefix, true));
            h = b.modulate(ggml_norm(ctx, joint, d->eps), modulation[2], prefix, false);
            ggml_tensor *gate = nullptr, *up = nullptr, *activated = nullptr;
            if (!w.up.data) {
                // A fused projection carries at most one update, sized for both halves.
                auto gu = b.linear(w.gate, h, l ? &l->gate : nullptr);
                // The checkpoint stores [gate, up] in one projection. Upstream
                // SwiGLU reads that layout directly, avoiding two FF-sized
                // copies and a materialized SiLU activation per layer.
                auto fused = ggml_swiglu(ctx, gu);
                if (backend_supports_op(fused)) activated = fused;
                else {
                    int ff = gu->ne[0] / 2;
                    gate = ggml_cont(ctx, ggml_view_2d(ctx, gu, ff, seq, gu->nb[1], 0));
                    up = ggml_cont(ctx, ggml_view_2d(ctx, gu, ff, seq, gu->nb[1], ff * sizeof(float)));
                }
            } else {
                const auto gu = l ? b.shrink({&l->gate, &l->up}, h) : std::vector<ggml_tensor*>(2, nullptr);
                gate = b.linear(w.gate, h, l ? &l->gate : nullptr, gu[0]);
                up = b.linear(w.up, h, l ? &l->up : nullptr, gu[1]);
            }
            if (!activated) {
                auto fused = ggml_swiglu_split(ctx, gate, up);
                activated = backend_supports_op(fused) ? fused : ggml_mul(ctx, up, ggml_silu(ctx, gate));
            }
            h = b.linear(w.down, activated, l ? &l->down : nullptr);
            if (d->tp_ranks > 1) result->boundaries.push_back(h);
            joint = ggml_add(ctx, joint, b.modulate(h, modulation[3], prefix, true));
        }
        auto target = b.slice(joint, prefix, seq - prefix);
        auto scale = ggml_scale_bias(ctx, b.linear(d->norm_out, b.slice(time, 0, 1), global(&TSGQi21Adapter::norm_out)), 1.f, 1.f);
        auto normalized = ggml_mul(ctx, ggml_norm(ctx, target, d->eps), scale);
        ggml_tensor* output;
        if (adapter && adapter->output_head) {
            // A per-call head replaces proj_out; its data is uploaded with the inputs.
            auto head = b.input(kHead, 0, d->dim, d->channels, static_cast<ggml_type>(adapter->output_head_type));
            output = ggml_mul_mat(ctx, head, normalized);
            ggml_prec_set_acc(output, GGML_PREC_F32);
        } else {
            output = b.linear(d->proj_out, normalized, global(&TSGQi21Adapter::proj_out));
        }
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        if (persistent) {
            // Gallocr may recycle an input after its last consumer. Constants
            // uploaded only at build time must survive every subsequent replay.
            for (const auto& c : b.constants) ggml_set_output(c.tensor);
            result->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
            if (!result->allocator) throw std::runtime_error("QwenImage21: graph allocator creation failed");
            size_t need = 0, free_bytes = 0, total_bytes = 0;
            ggml_gallocr_reserve_n_size(result->allocator, graph, nullptr, nullptr, &need);
            ggml_backend_dev_memory(ggml_backend_get_device(g_backend), &free_bytes, &total_bytes);
            // A second CFG shape is optional: retire the other slot when both
            // scratch arenas would overcommit VRAM and spill on Windows.
            if (total_bytes && (need > free_bytes || free_bytes - need < size_t(512) * 1024 * 1024)) {
                for (auto& retained : graph_cache[g_active_rank]) retained.reset();
            }
            // The size-only reserve populates plans without backing buffers;
            // alloc_graph cannot detect that case, so reserve storage explicitly.
            if (!ggml_gallocr_reserve(result->allocator, graph) || !ggml_gallocr_alloc_graph(result->allocator, graph))
                throw std::runtime_error("QwenImage21: persistent graph allocation failed");
        } else if (!alloc_graph_reuse_gallocr(graph)) {
            throw std::runtime_error("QwenImage21: graph allocation failed");
        }
        host_read_barrier();
        // Masks of a declined/disabled flash operation are not reachable from
        // the final graph and therefore have no allocator slot.
        for (const auto& c : b.constants)
            if (c.tensor->buffer) ggml_backend_tensor_set(c.tensor, c.data, 0, c.bytes);
        result->graph = graph;
        result->output = output;
        result->inputs = std::move(b.inputs);
        result->resident = std::move(b.resident);
        if (d->tp_ranks > 1) {
            result->plan.graph = graph;
            result->plan.ar_tensor = result->boundaries;
            if (!tp_plan_segments(result->plan, result->boundaries))
                throw std::runtime_error("QwenImage21: tensor-parallel segment plan failed");
            result->plan.out_tensor = output;
            result->plan.out_bytes = ggml_nbytes(output);
        }
        if (option_enabled("TS_QWEN21_GRAPH_TRACE", false)) {
            static const char* names[] = {"full", "extract", "cached"};
            std::fprintf(stderr, "[qwen21-graph] build mode=%s tokens=%d/%d segments=%d nodes=%d scratch=%.1f MiB\n",
                names[static_cast<int>(mode)], seq, d->total_seq, num_segments, ggml_graph_n_nodes(graph),
                result->allocator ? ggml_gallocr_get_buffer_size(result->allocator, 0) / (1024.0 * 1024.0) : 0.0);
        }
        return result;
}

void upload_inputs(ForwardGraph& entry, const TSGQi21Desc& d) {
    host_read_barrier();
    for (const auto& input : entry.inputs)
        if (input.tensor->buffer)
            ggml_backend_tensor_set(input.tensor, field_data(d, input.field, input.offset), 0, ggml_nbytes(input.tensor));
}

void run_graph(ForwardGraph& entry, const TSGQi21Desc& d) {
    upload_inputs(entry, d);
    if (compute_graph(g_backend, entry.graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("QwenImage21: graph compute failed");
    ggml_backend_tensor_get(entry.output, d.output, 0, ggml_nbytes(entry.output));
    ++entry.runs;
    entry.used = ++graph_use;
    if (entry.allocator && option_enabled("TS_QWEN21_GRAPH_TRACE", false))
        std::fprintf(stderr, "[qwen21-graph] run=%llu tokens=%d\n",
            static_cast<unsigned long long>(entry.runs), static_cast<int>(ggml_nelements(entry.output) / d.channels));
}

bool persistent_graphs() {
    return (g_backend_type == BACKEND_TYPE_CUDA || g_backend_type == BACKEND_TYPE_METAL) &&
        option_enabled("TS_QWEN21_GRAPH_REUSE", true);
}

// The retained graph for (mode, cache) on the active rank, built into the least
// recently used slot when absent.
ForwardGraph& retained_graph(const TSGQi21Desc& d, Mode mode, PrefixCache* cache) {
    auto& entries = graph_cache[g_active_rank];
    const auto key = weights(d);
    const uint64_t cache_id = cache ? cache->id : 0;
    for (auto& entry : entries) if (entry && entry->matches(d, key, mode, cache_id)) {
        entry->used = ++graph_use;
        return *entry;
    }
    auto slot = std::min_element(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
        return (!a ? 0 : a->used) < (!b ? 0 : b->used);
    });
    slot->reset();
    *slot = build_graph(&d, true, mode, cache);
    (*slot)->used = ++graph_use;
    return **slot;
}

// Runs a retained graph for (mode, cache). Without retention the graph is transient.
void run_retained(const TSGQi21Desc& d, Mode mode, PrefixCache* cache) {
    if (!persistent_graphs()) {
        // Disabling reuse also releases previous entries before allocating
        // baseline scratch, so A/B runs do not charge both sets of buffers.
        for (auto& entry : graph_cache[g_active_rank]) entry.reset();
        auto transient = build_graph(&d, false, mode, cache);
        run_graph(*transient, d);
        return;
    }
    run_graph(retained_graph(d, mode, cache), d);
}

ggml_type stored_type(ggml_type attn_type, std::int32_t requested, bool is_value) {
    switch (requested) {
        case TSG_QI21_PREFIX_F32: return GGML_TYPE_F32;
        case TSG_QI21_PREFIX_F16: return GGML_TYPE_F16;
        case TSG_QI21_PREFIX_Q8_0: return GGML_TYPE_Q8_0;
        // Attention logits are more sensitive to K than the output is to V.
        case TSG_QI21_PREFIX_Q8_0_V: return is_value ? GGML_TYPE_Q8_0 : attn_type;
        default: return attn_type;
    }
}

// Optional user cap on one cache, in MiB; unset means only the free-memory rule.
bool within_user_cap(size_t bytes) {
    const char* value = std::getenv("TS_QWEN21_PREFIX_CACHE_MAX_MIB");
    if (!value || !*value) return true;
    char* end = nullptr;
    const double mib = std::strtod(value, &end);
    return end != value && std::isfinite(mib) && mib >= 0 && static_cast<double>(bytes) <= mib * 1024.0 * 1024.0;
}

// Allocates the cache for a new key, or declines it with a warning when the
// device cannot hold it next to the step graphs.
std::unique_ptr<PrefixCache> create_prefix_cache(const TSGQi21Desc& d, const std::vector<TSGQi21Weight>& key) {
    auto cache = std::make_unique<PrefixCache>();
    cache->key = d.prefix_cache_key;
    cache->id = ++prefix_cache_ids;
    cache->rank = g_active_rank;
    cache->backend = g_backend;
    cache->shape = d;
    cache->segments.assign(d.segments, d.segments + d.num_segments);
    cache->weight_key = key;
    cache->flash = option_enabled("TS_QWEN21_FLASH", true);
    cache->pad_mask = option_enabled("TS_QWEN21_PAD_MASK", false);
    cache->requested_type = d.prefix_cache_type;
    cache->attn_type = attention_kv_type();
    cache->k_type = stored_type(cache->attn_type, d.prefix_cache_type, false);
    cache->v_type = stored_type(cache->attn_type, d.prefix_cache_type, true);
    const size_t per_layer = (ggml_row_size(cache->k_type, d.head_dim) + ggml_row_size(cache->v_type, d.head_dim)) *
        static_cast<size_t>(d.heads) * static_cast<size_t>(d.prefix_seq);
    cache->bytes = per_layer * static_cast<size_t>(d.num_layers);
    size_t free_bytes = 0, total_bytes = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(g_backend), &free_bytes, &total_bytes);
    // The prefix is at most half of what the device reports free, so the step
    // graphs' scratch still fits beside it.
    const bool fits = (!total_bytes || cache->bytes <= free_bytes / 2) && within_user_cap(cache->bytes);
    if (fits && d.prefix_seq > 0) {
        ggml_init_params init{ggml_tensor_overhead() * (2 * static_cast<size_t>(d.num_layers) + 8), nullptr, true};
        cache->ctx = ggml_init(init);
        if (!cache->ctx) throw std::runtime_error("QwenImage21: prefix cache context allocation failed");
        for (int i = 0; i < d.num_layers; ++i) {
            cache->k.push_back(ggml_new_tensor_3d(cache->ctx, cache->k_type, d.head_dim, d.prefix_seq, d.heads));
            cache->v.push_back(ggml_new_tensor_3d(cache->ctx, cache->v_type, d.head_dim, d.prefix_seq, d.heads));
        }
        cache->buffer = ggml_backend_alloc_ctx_tensors_from_buft(cache->ctx, ggml_backend_get_default_buffer_type(g_backend));
    }
    if (!cache->buffer) {
        cache->declined = true;
        std::fprintf(stderr,
            "[qwen21] prefix KV cache declined: %d prefix tokens need %.1f MiB (%s), device reports %.1f MiB free "
            "(the cache may use half; TS_QWEN21_PREFIX_CACHE_MAX_MIB caps it further); "
            "this request recomputes the prefix every step.\n",
            d.prefix_seq, cache->bytes / (1024.0 * 1024.0), ggml_type_name(cache->k_type), free_bytes / (1024.0 * 1024.0));
        if (cache->ctx) { ggml_free(cache->ctx); cache->ctx = nullptr; }
        cache->k.clear(); cache->v.clear();
    }
    return cache;
}

bool cache_describes(const PrefixCache& cache, const TSGQi21Desc& d, const std::vector<TSGQi21Weight>& key) {
    const auto& s = cache.shape;
    return cache.backend == g_backend && cache.rank == g_active_rank &&
        cache.requested_type == d.prefix_cache_type && s.dim == d.dim && s.heads == d.heads &&
        s.head_dim == d.head_dim && s.channels == d.channels && s.text_dim == d.text_dim &&
        s.image_seq == d.image_seq && s.text_seq == d.text_seq && s.total_seq == d.total_seq &&
        s.prefix_seq == d.prefix_seq && s.num_layers == d.num_layers && s.num_segments == d.num_segments &&
        s.tp_ranks == d.tp_ranks &&
        s.eps == d.eps && cache.flash == option_enabled("TS_QWEN21_FLASH", true) &&
        cache.pad_mask == option_enabled("TS_QWEN21_PAD_MASK", false) &&
        cache.attn_type == attention_kv_type() &&
        same_segments(cache.segments.data(), d.segments, d.num_segments) && same_weights(cache.weight_key, key);
}

// The active rank's cache for the descriptor's key, created on first use.
PrefixCache& prefix_cache_for(const TSGQi21Desc& d) {
    const auto key = weights(d);
    auto it = std::find_if(prefix_caches.begin(), prefix_caches.end(), [&](const auto& c) {
        return c->key == d.prefix_cache_key && c->rank == g_active_rank;
    });
    // A key describes one request's prefix; anything that changes its layout,
    // weights or attention configuration retires the stored values.
    if (it != prefix_caches.end() && !cache_describes(**it, d, key)) { drop_prefix_cache(it); it = prefix_caches.end(); }
    if (it == prefix_caches.end()) {
        // A tensor-parallel request holds one cache per rank and branch.
        while (prefix_caches.size() >= kMaxPrefixCaches * static_cast<size_t>(std::max(1, d.tp_ranks))) {
            auto oldest = std::min_element(prefix_caches.begin(), prefix_caches.end(),
                [](const auto& a, const auto& b) { return a->used < b->used; });
            drop_prefix_cache(oldest);
        }
        prefix_caches.push_back(create_prefix_cache(d, key));
        it = prefix_caches.end() - 1;
    }
    (*it)->used = ++prefix_cache_use;
    return **it;
}

int forward_with_prefix_cache(const TSGQi21Desc& d) {
    PrefixCache& cache = prefix_cache_for(d);
    if (cache.declined) { run_retained(d, Mode::Full, nullptr); return TSG_QI21_PATH_DECLINED; }
    if (cache.filled) { run_retained(d, Mode::Cached, &cache); return TSG_QI21_PATH_CACHED; }
    // The extract graph runs once per request, so it is never retained: its
    // whole-sequence scratch is released as soon as the prefix is stored.
    {
        auto extract = build_graph(&d, persistent_graphs(), Mode::Extract, &cache);
        run_graph(*extract, d);
    }
    cache.filled = true;
    return TSG_QI21_PATH_EXTRACT;
}
}

namespace tsg {
void qwen_image21_invalidate_weight(const void* data) {
    std::lock_guard<std::recursive_mutex> lock(graph_mutex);
    for (auto& device : graph_cache) for (auto& entry : device) {
        if (entry && std::any_of(entry->weight_key.begin(), entry->weight_key.end(),
            [data](const TSGQi21Weight& w) { return w.data == data; })) entry.reset();
    }
    // Stored K/V were computed from the old weights.
    for (auto it = prefix_caches.begin(); it != prefix_caches.end();) {
        if (std::any_of((*it)->weight_key.begin(), (*it)->weight_key.end(),
            [data](const TSGQi21Weight& w) { return w.data == data; })) {
            drop_graphs_of_cache((*it)->id);
            it = prefix_caches.erase(it);
        } else ++it;
    }
}
}

// Retires every retained graph. Stored prefixes survive a scratch release; a
// request that finds its cache gone simply stores it again.
TSG_EXPORT void TSGgml_QwenImage21ResetForwardCache() {
    std::lock_guard<std::recursive_mutex> lock(graph_mutex);
    for (auto& device : graph_cache) for (auto& entry : device) entry.reset();
}

TSG_EXPORT void TSGgml_QwenImage21ReleasePrefixCache(std::uint64_t key) {
    std::lock_guard<std::recursive_mutex> lock(graph_mutex);
    host_read_barrier();
    for (auto it = prefix_caches.begin(); it != prefix_caches.end();) {
        if ((*it)->key == key) {
            drop_graphs_of_cache((*it)->id);
            it = prefix_caches.erase(it);
        } else ++it;
    }
}

TSG_EXPORT void TSGgml_QwenImage21ReleasePrefixCaches() {
    std::lock_guard<std::recursive_mutex> lock(graph_mutex);
    host_read_barrier();
    for (const auto& cache : prefix_caches) drop_graphs_of_cache(cache->id);
    prefix_caches.clear();
}

TSG_EXPORT int TSGgml_QwenImage21GetPrefixCacheInfo(std::uint64_t key, TSGQi21PrefixCacheInfo* info) {
    std::lock_guard<std::recursive_mutex> lock(graph_mutex);
    if (!info) return 0;
    *info = {};
    for (const auto& cache : prefix_caches) {
        if (cache->key != key || cache->rank != g_active_rank) continue;
        info->state = cache->declined ? 2 : cache->filled ? 1 : 0;
        info->key_type = cache->k_type;
        info->value_type = cache->v_type;
        info->tokens = cache->shape.prefix_seq;
        info->bytes = static_cast<std::int64_t>(cache->bytes);
        return 1;
    }
    return 1;
}

TSG_EXPORT int TSGgml_QwenImage21Forward(const TSGQi21Desc* d) {
    std::lock_guard<std::recursive_mutex> lock(graph_mutex);
    try {
        validate(d);
        if (d->tp_ranks > 1) throw std::invalid_argument("QwenImage21: sharded weights require TSGgml_QwenImage21ForwardTp");
        if (!ensure_backend()) return 0;
        int path = TSG_QI21_PATH_FULL;
        // A request without a prefix (zero-length) has nothing to store.
        if (d->prefix_cache_key != 0 && d->prefix_seq > 0) path = forward_with_prefix_cache(*d);
        else run_retained(*d, Mode::Full, nullptr);
        clear_last_error();
        return path;
    } catch (const std::exception& e) {
        TSGgml_QwenImage21ResetForwardCache(); TSGgml_QwenImage21ReleasePrefixCaches();
        set_last_error(e.what()); return 0;
    } catch (...) {
        TSGgml_QwenImage21ResetForwardCache(); TSGgml_QwenImage21ReleasePrefixCaches();
        set_last_error("QwenImage21: unknown native error"); return 0;
    }
}

// One velocity prediction over a tensor-parallel group: descs[r] carries rank r's
// sharded block weights and the same inputs. Each rank runs its own graph; the
// shared segment executor submits every rank's segment, all-reduces the partial
// sums at each boundary (on-device through NCCL/P2P when ggml-cuda provides the
// collective, otherwise through host staging) and continues. Rank 0's output is
// the prediction. Each rank caches the prefix K/V of its own heads.
TSG_EXPORT int TSGgml_QwenImage21ForwardTp(const TSGQi21Desc* const* descs, int ranks) {
    std::lock_guard<std::recursive_mutex> lock(graph_mutex);
    try {
        if (!descs || ranks < 2 || ranks > TSG_MAX_DEVICES)
            throw std::invalid_argument("QwenImage21: invalid tensor-parallel rank count");
        for (int r = 0; r < ranks; ++r) {
            validate(descs[r]);
            const auto& a = *descs[0];
            const auto& d = *descs[r];
            if (d.tp_ranks != ranks || d.dim != a.dim || d.heads != a.heads || d.head_dim != a.head_dim ||
                d.channels != a.channels || d.text_dim != a.text_dim || d.image_seq != a.image_seq ||
                d.text_seq != a.text_seq || d.total_seq != a.total_seq || d.prefix_seq != a.prefix_seq ||
                d.num_layers != a.num_layers || d.num_segments != a.num_segments || d.eps != a.eps ||
                d.prefix_cache_key != a.prefix_cache_key || d.prefix_cache_type != a.prefix_cache_type ||
                (d.adapter == nullptr) != (a.adapter == nullptr) ||
                (d.adapter && (d.adapter->output_head == nullptr) != (a.adapter->output_head == nullptr)) ||
                !same_segments(d.segments, a.segments, a.num_segments))
                throw std::invalid_argument("QwenImage21: tensor-parallel ranks disagree on the forward shape");
        }
        if (!ensure_backend()) return 0;
        if (!tp_fused_available(ranks))
            throw std::runtime_error("QwenImage21: no initialized tensor-parallel group of " + std::to_string(ranks) + " ranks");

        // One decision for every rank: a cache stored on some ranks but declined
        // on another would leave the ranks computing different sequences.
        Mode mode = Mode::Full;
        int path = TSG_QI21_PATH_FULL;
        std::vector<PrefixCache*> caches(static_cast<size_t>(ranks), nullptr);
        if (descs[0]->prefix_cache_key != 0 && descs[0]->prefix_seq > 0) {
            bool declined = false, filled = true;
            for (int r = 0; r < ranks; ++r) {
                ScopedRank rank(r);
                caches[r] = &prefix_cache_for(*descs[r]);
                declined |= caches[r]->declined;
                filled &= caches[r]->filled;
            }
            if (declined) {
                for (auto* cache : caches) {
                    if (cache->declined) continue;
                    drop_graphs_of_cache(cache->id);
                    cache->declined = true; cache->filled = false;
                    cache->k.clear(); cache->v.clear();
                    if (cache->buffer) { ggml_backend_buffer_free(cache->buffer); cache->buffer = nullptr; }
                }
                std::fill(caches.begin(), caches.end(), nullptr);
                path = TSG_QI21_PATH_DECLINED;
            } else {
                mode = filled ? Mode::Cached : Mode::Extract;
                path = filled ? TSG_QI21_PATH_CACHED : TSG_QI21_PATH_EXTRACT;
            }
        }

        // The extract graph runs once per request and is not retained; neither is
        // anything when retention is disabled.
        const bool retain = persistent_graphs() && mode != Mode::Extract;
        std::vector<std::unique_ptr<ForwardGraph>> transient(static_cast<size_t>(ranks));
        std::vector<TpRankPlan*> plans(static_cast<size_t>(ranks), nullptr);
        for (int r = 0; r < ranks; ++r) {
            ScopedRank rank(r);
            ForwardGraph* graph = nullptr;
            if (retain) graph = &retained_graph(*descs[r], mode, caches[r]);
            else {
                if (!persistent_graphs()) for (auto& entry : graph_cache[r]) entry.reset();
                transient[r] = build_graph(descs[r], persistent_graphs(), mode, caches[r]);
                graph = transient[r].get();
            }
            upload_inputs(*graph, *descs[r]);
            graph->plan.out_host = r == 0 ? descs[0]->output : nullptr;
            graph->plan.out_bytes = r == 0 ? ggml_nbytes(graph->output) : 0;
            plans[r] = &graph->plan;
            ++graph->runs;
        }
        if (!tp_execute_plans(plans.data(), ranks))
            throw std::runtime_error("QwenImage21: tensor-parallel execution failed: " + g_last_error);
        if (mode == Mode::Extract) for (auto* cache : caches) cache->filled = true;
        clear_last_error();
        return path;
    } catch (const std::exception& e) {
        TSGgml_QwenImage21ResetForwardCache(); TSGgml_QwenImage21ReleasePrefixCaches();
        set_last_error(e.what()); return 0;
    } catch (...) {
        TSGgml_QwenImage21ResetForwardCache(); TSGgml_QwenImage21ReleasePrefixCaches();
        set_last_error("QwenImage21: unknown native error"); return 0;
    }
}
