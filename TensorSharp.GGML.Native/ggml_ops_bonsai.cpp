// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_internal.h"
#include "ggml_ops_bonsai.h"
#include "ggml_ops_q8_weight_policy.h"
#include <map>
#include <tuple>

namespace {
struct DeviceConstants {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * rotation = nullptr;
    ggml_tensor * signs = nullptr;
    ~DeviceConstants() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
};
struct Transform {
    int width, block, hd, nk, rep;
    bool inverse;
    std::vector<float> signs;
    std::map<ggml_backend_t, std::unique_ptr<DeviceConstants>> devices;
};
std::mutex registry_mutex;
std::unordered_map<const void *, std::shared_ptr<Transform>> registry;
std::atomic<bool> has_transforms{false};

std::shared_ptr<Transform> lookup(const void * key) {
    if (!has_transforms.load(std::memory_order_acquire)) return {};
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it = registry.find(key);
    return it == registry.end() ? nullptr : it->second;
}

DeviceConstants & constants(Transform & t) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto & ptr = t.devices[tsg::active_backend()];
    if (ptr) return *ptr;
    auto c = std::make_unique<DeviceConstants>();
    ggml_init_params p{ggml_tensor_overhead() * 2 + 1024, nullptr, true};
    c->ctx = ggml_init(p);
    if (!c->ctx) throw std::runtime_error("Bonsai: cannot allocate transform context");
    // Upstream Metal has fast FWHT kernels through 512. Factor H1024 into
    // two H512 transforms and one butterfly instead of a dense 1024 GEMM.
    const int n = std::min(t.block, 512);
    c->rotation = ggml_new_tensor_2d(c->ctx, GGML_TYPE_F32, n, n);
    c->signs = ggml_new_tensor_1d(c->ctx, GGML_TYPE_F32, t.width);
    ggml_set_name(c->rotation, "bonsai.normalized_hadamard");
    ggml_set_name(c->signs, "bonsai.signs");
    c->buffer = ggml_backend_alloc_ctx_tensors(c->ctx, tsg::active_backend());
    if (!c->buffer) throw std::runtime_error("Bonsai: cannot allocate transform constants");
    ggml_backend_buffer_set_usage(c->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> h(size_t(n) * n);
    const float scale = 1.0f / std::sqrt(float(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            unsigned bits = unsigned(i & j);
            bits ^= bits >> 16; bits ^= bits >> 8; bits ^= bits >> 4;
            bits ^= bits >> 2; bits ^= bits >> 1;
            h[size_t(i) * n + j] = bits & 1 ? -scale : scale;
        }
    ggml_backend_tensor_set(c->rotation, h.data(), 0, h.size() * sizeof(float));
    // H512 is normalized by the upstream FWHT kernel. H1024's final
    // butterfly needs one more 1/sqrt(2): fold it into the mandatory sign
    // multiplication (before forward FWHT, after inverse FWHT) instead of
    // dispatching a separate scale kernel for every rotated projection.
    std::vector<float> signed_normalization = t.signs;
    if (t.block == 1024)
        for (float & sign : signed_normalization) sign *= 0.7071067811865475244f;
    ggml_backend_tensor_set(c->signs, signed_normalization.data(), 0,
                           signed_normalization.size() * sizeof(float));
    ptr = std::move(c);
    return *ptr;
}

ggml_tensor * fwht(ggml_context * ctx, ggml_tensor * input, const Transform & t, DeviceConstants & c) {
    auto * flat = ggml_is_contiguous(input) ? input : ggml_cont(ctx, input);
    // The mul_mat below is tagged GGML_HINT_SRC0_IS_HADAMARD, which asks the backend to
    // ignore c.rotation and run its own FWHT over src1 instead. ggml-cpu type-checks src1
    // before taking that path; ggml-cuda's fwht kernel does not, and reads an F16 row as
    // floats - wrong values, and twice the bytes. That is how the inverse direction
    // produced 16.0 max abs error against the dense oracle on an A5000 (bonsai2-hadamard-cuda)
    // while the CPU backend was correct. The scheduler's supports_op would have declined the
    // op, but these graphs go straight to ggml_backend_graph_compute, so nothing catches it.
    // Widening here covers every caller: the forward direction already widens at its own call
    // site, for the unrelated reason that F16 would round the folded 1/sqrt(2) normalization.
    // Model activations and get_rows output are already F32, so no path the model runs gains
    // a node - this is a guard for the contract, not a cost.
    if (flat->type != GGML_TYPE_F32) flat = ggml_cast(ctx, flat, GGML_TYPE_F32);
    const int64_t count = ggml_nelements(input);
    auto * x = ggml_reshape_2d(ctx, flat, c.rotation->ne[0], count / c.rotation->ne[0]);
    x = ggml_mul_mat(ctx, c.rotation, x);
    ggml_mul_mat_set_hint(x, GGML_HINT_SRC0_IS_HADAMARD);
    if (t.block == 1024) {
        const size_t stride = 1024 * sizeof(float);
        auto * a = ggml_view_2d(ctx, x, 512, count / 1024, stride, 0);
        auto * b = ggml_view_2d(ctx, x, 512, count / 1024, stride, 512 * sizeof(float));
        auto * plus = ggml_add(ctx, a, b);
        auto * minus = ggml_sub(ctx, a, b);
        x = ggml_concat(ctx, plus, minus, 0);
    }
    return ggml_reshape_4d(ctx, x, input->ne[0], input->ne[1], input->ne[2], input->ne[3]);
}
}

namespace tsg {
bool bonsai_weight_has_transform(const void * key) {
    return bool(lookup(key));
}
struct BonsaiGraphScope::Impl {
    Impl * previous;
    std::map<std::tuple<const ggml_context *, const ggml_tensor *, const Transform *>, ggml_tensor *> memo;
};
static thread_local BonsaiGraphScope::Impl * current_scope = nullptr;
BonsaiGraphScope::BonsaiGraphScope() : impl(new Impl{current_scope, {}}) { current_scope = impl.get(); }
BonsaiGraphScope::~BonsaiGraphScope() { current_scope = impl->previous; }

ggml_tensor * bonsai_transform(ggml_context * ctx, ggml_tensor * x, const void * key, bool inverse) {
    auto t = lookup(key);
    if (!t) return x;
    if (t->inverse != inverse || x->ne[0] != t->width)
        throw std::runtime_error("Bonsai: weight transform direction or input width mismatch");
    const auto memo_key = std::make_tuple(static_cast<const ggml_context *>(ctx), static_cast<const ggml_tensor *>(x), t.get());
    if (current_scope) {
        auto found = current_scope->memo.find(memo_key);
        if (found != current_scope->memo.end()) return found->second;
    }
    auto & c = constants(*t);
    if (inverse) {
        x = ggml_mul(ctx, fwht(ctx, x, *t, c), c.signs);
    } else {
        if (t->rep > 1) {
            const int64_t n1 = x->ne[1], n2 = x->ne[2], n3 = x->ne[3];
            x = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
            x = ggml_reshape_4d(ctx, x, t->hd, t->nk, t->rep, n1*n2*n3);
            x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
            x = ggml_reshape_4d(ctx, x, t->width, n1, n2, n3);
        }
        // A half input must be widened before the folded normalization:
        // multiplying ±1/sqrt(2) in F16 would round before the FWHT, unlike
        // the original post-transform F32 scale. Normal model activations
        // are already F32, so this compatibility path adds no decode node.
        if (x->type != GGML_TYPE_F32) x = ggml_cast(ctx, x, GGML_TYPE_F32);
        x = fwht(ctx, ggml_mul(ctx, x, c.signs), *t, c);
    }
    ggml_set_name(x, inverse ? "bonsai.inverse_embedding" : "bonsai.activation");
    if (current_scope) current_scope->memo.emplace(memo_key, x);
    return x;
}
ggml_tensor * bonsai_mul_mat(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, const void * key) {
    return tsg::weight_mul_mat(ctx, w, bonsai_transform(ctx, x, key, false), key);
}
ggml_tensor * bonsai_get_rows(ggml_context * ctx, ggml_tensor * w, ggml_tensor * ids, const void * key) {
    return bonsai_transform(ctx, ggml_get_rows(ctx, w, ids), key, true);
}
void bonsai_clear_backend() {
    std::lock_guard<std::mutex> lock(registry_mutex);
    for (auto & entry : registry) entry.second->devices.clear();
}
}

TSG_EXPORT int TSGgml_BonsaiRegisterWeight(const void * key, int width, const float * signs,
        int block, int inverse, int hd, int nk, int rep) {
    try {
        if (!key || !signs || width <= 0 || (block != 64 && block != 128 && block != 256 && block != 512 && block != 1024)
            || width % block || (inverse != 0 && inverse != 1) || rep < 1
            || (rep > 1 && (inverse || hd <= 0 || nk <= 0 || int64_t(hd)*nk*rep != width)))
            throw std::runtime_error("Bonsai: invalid weight transform parameters");
        for (int i = 0; i < width; ++i)
            if (signs[i] != 1.0f && signs[i] != -1.0f)
                throw std::runtime_error("Bonsai: transform signs must be +1 or -1");
        std::lock_guard<std::mutex> lock(registry_mutex);
        if (registry.count(key)) throw std::runtime_error("Bonsai: weight already registered");
        std::shared_ptr<Transform> t;
        for (auto & entry : registry) {
            auto & candidate = entry.second;
            if (candidate->width == width && candidate->block == block && candidate->inverse == bool(inverse)
                && candidate->hd == hd && candidate->nk == nk && candidate->rep == rep
                && std::equal(candidate->signs.begin(), candidate->signs.end(), signs)) {
                t = candidate; break;
            }
        }
        if (!t) {
            t = std::make_shared<Transform>();
            t->width = width; t->block = block; t->inverse = inverse;
            t->hd = hd; t->nk = nk; t->rep = rep;
            t->signs.assign(signs, signs + width);
        }
        registry.emplace(key, std::move(t));
        has_transforms.store(true, std::memory_order_release);
        return 1;
    } catch (const std::exception & e) { tsg::set_last_error(e.what()); return 0; }
}
TSG_EXPORT void TSGgml_BonsaiUnregisterWeight(const void * key) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    registry.erase(key);
    has_transforms.store(!registry.empty(), std::memory_order_release);
}
