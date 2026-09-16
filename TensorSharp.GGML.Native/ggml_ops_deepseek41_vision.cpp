// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_deepseek41_vision.h"
#include "ggml_ops_internal.h"
#include "gguf.h"
#include "ggml_ops_dsv4_fused.h"
#include "ggml_ops_matmul_precision.h"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <utility>

namespace tsg_dsv41_vision {
namespace {
void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error("DeepSeek V4.1 vision: " + message);
}
struct file_context {
    gguf_context * file = nullptr;
    ggml_context * tensors = nullptr;
    ~file_context() { if (file) gguf_free(file); if (tensors) ggml_free(tensors); }
};
int read_int(const gguf_context * file, const std::string & name) {
    const int64_t key = gguf_find_key(file, name.c_str());
    require(key >= 0, "missing metadata " + name);
    uint64_t value = UINT64_MAX;
    switch (gguf_get_kv_type(file, key)) {
        case GGUF_TYPE_UINT32: value = gguf_get_val_u32(file, key); break;
        case GGUF_TYPE_UINT64: value = gguf_get_val_u64(file, key); break;
        case GGUF_TYPE_INT32: value = uint64_t(gguf_get_val_i32(file, key)); break;
        case GGUF_TYPE_INT64: value = uint64_t(gguf_get_val_i64(file, key)); break;
        default: break;
    }
    require(value <= INT32_MAX, "invalid metadata " + name);
    return int(value);
}
std::vector<float> read_vector(std::ifstream & input, const gguf_context * file,
                                ggml_context * tensors, const std::string & name) {
    auto * tensor = ggml_get_tensor(tensors, name.c_str());
    require(tensor && ggml_is_vector(tensor), "missing vector " + name);
    const int64_t index = gguf_find_tensor(file, name.c_str());
    input.seekg(gguf_get_data_offset(file) + gguf_get_tensor_offset(file, index));
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    require(bool(input.read(reinterpret_cast<char *>(bytes.data()), bytes.size())), "cannot read " + name);
    std::vector<float> values(ggml_nelements(tensor));
    if (tensor->type == GGML_TYPE_F32) std::memcpy(values.data(), bytes.data(), bytes.size());
    else {
        const auto * traits = ggml_get_type_traits(tensor->type);
        require(traits->to_float != nullptr, "unsupported vector type for " + name);
        traits->to_float(bytes.data(), values.data(), values.size());
    }
    for (float value : values) require(std::isfinite(value), "nonfinite vector " + name);
    return values;
}
} // namespace

struct encoder::implementation {
    metadata meta;
    ggml_backend_t backend = nullptr;
    ggml_backend_t cuda_backend = nullptr;
    ggml_context * weights_ctx = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    std::map<std::string, ggml_tensor *> weights;
    std::vector<std::vector<float>> biases;
    std::vector<float> start, newline, end;
    std::mutex mutex;

    struct graph {
        ggml_context * ctx = nullptr;
        ggml_gallocr_t allocator = nullptr;
        ggml_cgraph * gf = nullptr;
        ggml_tensor * patches = nullptr, * cos = nullptr, * sin = nullptr, * gather = nullptr;
        ggml_tensor * zero = nullptr, * output = nullptr;
        std::vector<std::pair<std::string, ggml_tensor *>> traces;
        bool tracing = false;
        int height = 0, width = 0;
        ~graph() { if (allocator) ggml_gallocr_free(allocator); if (ctx) ggml_free(ctx); }
    };
    std::unique_ptr<graph> cached;

    ~implementation() {
        cached.reset();
        if (weights_buffer) ggml_backend_buffer_free(weights_buffer);
        if (weights_ctx) ggml_free(weights_ctx);
        if (backend) ggml_backend_free(backend);
        if (cuda_backend) ggml_backend_free(cuda_backend);
    }

    ggml_tensor * weight(const std::string & name, int64_t ne0, int64_t ne1 = 1) {
        const auto it = weights.find(name);
        require(it != weights.end(), "missing tensor " + name);
        auto * value = it->second;
        require(value->ne[0] == ne0 && value->ne[1] == ne1 && value->ne[2] == 1 && value->ne[3] == 1,
                "unexpected tensor dimensions for " + name);
        return value;
    }

    void validate_weights() {
        const auto & m = meta;
        auto matrix = [&](const std::string & prefix, int input, int output, bool bias) {
            weight(prefix + ".weight", input, output);
            if (bias) weight(prefix + ".bias", output);
        };
        matrix("vision.patch_embed.proj", 3 * m.patch_size * m.patch_size, m.dim, true);
        for (int layer = 0; layer < m.layers; ++layer) {
            const std::string p = "vision.blocks." + std::to_string(layer);
            weight(p + ".norm1.weight", m.dim);
            weight(p + ".norm2.weight", m.dim);
            matrix(p + ".attn.wqkv", m.dim, 3 * m.dim, true);
            matrix(p + ".attn.wo", m.dim, m.dim, true);
            matrix(p + ".mlp.w1", m.dim, 2 * m.intermediate, false);
            matrix(p + ".mlp.w2", m.intermediate, m.dim, false);
        }
        weight("vision.norm.weight", m.dim);
        matrix("aligner.w1", m.dim * m.downsample_ratio * m.downsample_ratio, m.text_dim, true);
        matrix("aligner.w2", m.text_dim, m.text_dim, true);
    }

    ggml_tensor * round(ggml_context * ctx, ggml_tensor * value) const {
        return meta.bf16_activations ? ggml_cast(ctx, ggml_cast(ctx, value, GGML_TYPE_BF16), GGML_TYPE_F32) : value;
    }
    ggml_tensor * f32(ggml_context * ctx, ggml_tensor * value) const {
        return value->type == GGML_TYPE_F32 ? value : ggml_cast(ctx, value, GGML_TYPE_F32);
    }
    void precise(ggml_context * ctx, ggml_tensor * value) const {
        const bool cpu_needs_conversion = ggml_backend_is_cpu(backend) &&
            (value->src[0]->type != GGML_TYPE_F32 || value->src[1]->type != GGML_TYPE_F32);
        if (cuda_backend || cpu_needs_conversion) tsg_matmul_require_f32(ctx, value);
        else {
            // CPU F32/F32 keeps its native batching/reduction order. Other
            // GPU backends retain their existing public precision contract;
            // direct execution has no CPU scheduler for their custom nodes.
            ggml_prec_set_acc(value, GGML_PREC_F32);
            ggml_prec_set_src(value, GGML_PREC_F32, 1);
        }
    }
    ggml_tensor * linear(ggml_context * ctx, ggml_tensor * value, const std::string & prefix,
                         int input, int output, bool bias) {
        auto * matrix = weight(prefix + ".weight", input, output);
        auto * result = ggml_mul_mat(ctx, matrix, value);
        bool native_bf16_gemm = false;
#if defined(TSG_GGML_USE_CUDA)
        // NVIDIA's BF16 GEMM already accumulates and returns F32. Keep that
        // path so bias is added before the single BF16 rounding, without an
        // unnecessary conversion of both inputs to F32/TF32 GEMM. Other
        // backends retain explicit F32 until their output precision is tested.
        native_bf16_gemm = tsg_dsv4_cuda_supports_native_bf16(cuda_backend ? cuda_backend : backend);
#endif
        if (const char * setting = std::getenv("TS_DSV41_VISION_BF16_GEMM"))
            native_bf16_gemm = native_bf16_gemm && std::atoi(setting) != 0;
        if (matrix->type != GGML_TYPE_BF16 || !native_bf16_gemm)
            ggml_prec_set_acc(result, GGML_PREC_F32);
        // Synthetic F32 fixtures also protect against TF32 source truncation.
        if (matrix->type == GGML_TYPE_F32) precise(ctx, result);
        if (bias) result = ggml_add(ctx, result, f32(ctx, weight(prefix + ".bias", output)));
        return round(ctx, result);
    }
    ggml_tensor * norm(ggml_context * ctx, ggml_tensor * value, const std::string & name) {
        return round(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, value, 1e-6f), f32(ctx, weight(name, meta.dim))));
    }
    ggml_tensor * rotary(graph & g, ggml_tensor * value) const {
        const int half = meta.dim / meta.heads / 2, n = g.height * g.width;
        auto * first = ggml_view_3d(g.ctx, value, half, meta.heads, n, value->nb[1], value->nb[2], 0);
        auto * second = ggml_view_3d(g.ctx, value, half, meta.heads, n, value->nb[1], value->nb[2], half * sizeof(float));
        auto * a = ggml_sub(g.ctx, ggml_mul(g.ctx, first, g.cos), ggml_mul(g.ctx, second, g.sin));
        auto * b = ggml_add(g.ctx, ggml_mul(g.ctx, second, g.cos), ggml_mul(g.ctx, first, g.sin));
        return round(g.ctx, ggml_concat(g.ctx, a, b, 0));
    }

    std::unique_ptr<graph> build(int height, int width) {
        auto g = std::make_unique<graph>();
        g->height = height; g->width = width;
        g->tracing = std::getenv("TS_DSV41_VISION_TRACE_DIR") != nullptr;
        const int n = height * width, dim = meta.dim, head = dim / meta.heads;
        const int patch_dim = 3 * meta.patch_size * meta.patch_size;
        const int ratio = meta.downsample_ratio;
        const int merged_h = (height + ratio - 1) / ratio, merged_w = (width + ratio - 1) / ratio;
        const int merged = merged_h * merged_w;
        g->ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
        require(g->ctx != nullptr, "cannot allocate graph metadata");
        auto * ctx = g->ctx;
        auto trace = [&](const std::string & name, ggml_tensor * value) {
            if (!g->tracing) return;
            if (!ggml_is_contiguous(value)) value = ggml_cont(ctx, value);
            ggml_set_output(value);
            g->traces.emplace_back(name, value);
        };
        g->gf = ggml_new_graph_custom(ctx, 16384, false);
        g->patches = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, patch_dim, n);
        g->cos = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head / 2, 1, n);
        g->sin = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head / 2, 1, n);
        g->zero = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 1);
        g->gather = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ratio * ratio * merged);
        for (auto * input : {g->patches, g->cos, g->sin, g->zero, g->gather}) ggml_set_input(input);
        ggml_set_name(g->patches, "vision.patches");
        auto * x = linear(ctx, round(ctx, g->patches), "vision.patch_embed.proj", patch_dim, dim, true);
        trace("vision.patch_embed", x);
        // The official 3D SDPA call selects F32 math attention. ggml's flash
        // kernels introduce F16 intermediate rounding; keep that faster path
        // explicit rather than silently changing BF16 image features.
        bool use_fa = false;
        if (const char * setting = std::getenv("TS_DSV41_VISION_FA")) use_fa = std::atoi(setting) != 0;
        for (int layer = 0; layer < meta.layers; ++layer) {
            const std::string prefix = "vision.blocks." + std::to_string(layer);
            auto * residual = x;
            auto * normalized = norm(ctx, x, prefix + ".norm1.weight");
            auto * qkv = linear(ctx, normalized, prefix + ".attn.wqkv", dim, 3 * dim, true);
            if (layer == 0) { trace(prefix + ".norm1", normalized); trace(prefix + ".attn.wqkv", qkv); }
            auto view = [&](int index) {
                return ggml_view_3d(ctx, qkv, head, meta.heads, n,
                    head * sizeof(float), qkv->nb[1], index * dim * sizeof(float));
            };
            auto * q = ggml_permute(ctx, rotary(*g, view(0)), 0, 2, 1, 3);
            auto * k = ggml_permute(ctx, rotary(*g, view(1)), 0, 2, 1, 3);
            auto * v = ggml_permute(ctx, view(2), 0, 2, 1, 3);
            ggml_tensor * attention = nullptr;
            if (use_fa) {
                auto * kh = ggml_cast(ctx, ggml_cont(ctx, k), GGML_TYPE_F16);
                auto * vh = ggml_cast(ctx, ggml_cont(ctx, v), GGML_TYPE_F16);
                auto * fa = ggml_flash_attn_ext(ctx, q, kh, vh, nullptr, 1.0f / std::sqrt(float(head)), 0.0f, 0.0f);
                ggml_prec_set_acc(fa, GGML_PREC_F32);
                if (ggml_backend_supports_op(backend, fa)) attention = ggml_reshape_2d(ctx, fa, dim, n);
            }
            if (!attention) {
                auto * scores = ggml_mul_mat(ctx, k, q);
                precise(ctx, scores);
                scores = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f / std::sqrt(float(head)), 0.0f);
                auto * vt = ggml_cont(ctx, ggml_transpose(ctx, v));
                auto * out = ggml_mul_mat(ctx, vt, scores);
                precise(ctx, out);
                attention = ggml_cont_2d(ctx, ggml_permute(ctx, out, 0, 2, 1, 3), dim, n);
            }
            attention = round(ctx, attention);
            auto * projected = linear(ctx, attention, prefix + ".attn.wo", dim, dim, true);
            if (layer == 0) { trace(prefix + ".attn.context", attention); trace(prefix + ".attn.wo", projected); trace(prefix + ".attn", projected); }
            x = round(ctx, ggml_add(ctx, residual, projected));
            residual = x;
            normalized = norm(ctx, x, prefix + ".norm2.weight");
            auto * up_gate = linear(ctx, normalized, prefix + ".mlp.w1", dim, 2 * meta.intermediate, false);
            if (layer == 0) { trace(prefix + ".norm2", normalized); trace(prefix + ".mlp.w1", up_gate); }
            auto * gate = ggml_view_2d(ctx, up_gate, meta.intermediate, n, up_gate->nb[1], 0);
            auto * up = ggml_view_2d(ctx, up_gate, meta.intermediate, n, up_gate->nb[1], meta.intermediate * sizeof(float));
            auto * hidden = round(ctx, ggml_mul(ctx, round(ctx, ggml_silu(ctx, ggml_cont(ctx, gate))), up));
            projected = linear(ctx, hidden, prefix + ".mlp.w2", meta.intermediate, dim, false);
            if (layer == 0) { trace(prefix + ".mlp.w2", projected); trace(prefix + ".mlp", projected); }
            x = round(ctx, ggml_add(ctx, residual, projected));
            trace(prefix, x);
        }
        x = norm(ctx, x, "vision.norm.weight");
        trace("vision.norm", x);
        // F.unfold orders each merged column as channel,dy,dx. Padding is
        // zero after the final norm, not an additional image patch through ViT.
        x = ggml_concat(ctx, x, g->zero, 1);
        x = ggml_get_rows(ctx, x, g->gather);
        x = ggml_reshape_3d(ctx, x, dim, ratio * ratio, merged);
        x = ggml_cont_2d(ctx, ggml_permute(ctx, x, 1, 0, 2, 3), dim * ratio * ratio, merged);
        x = linear(ctx, x, "aligner.w1", dim * ratio * ratio, meta.text_dim, true);
        trace("aligner.w1", x);
        x = round(ctx, ggml_gelu_erf(ctx, x));
        g->output = linear(ctx, x, "aligner.w2", meta.text_dim, meta.text_dim, true);
        trace("aligner.w2", g->output);
        ggml_set_name(g->output, "vision.aligned");
        ggml_set_output(g->output);
        ggml_build_forward_expand(g->gf, g->output);
        for (const auto & entry : g->traces) ggml_build_forward_expand(g->gf, entry.second);
        for (int i = 0; i < ggml_graph_n_nodes(g->gf); ++i)
            require(ggml_backend_supports_op(backend, ggml_graph_node(g->gf, i)),
                    "backend does not support " + std::string(ggml_op_name(ggml_graph_node(g->gf, i)->op)));
        g->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(g->allocator && ggml_gallocr_alloc_graph(g->allocator, g->gf), "cannot allocate vision graph");
        return g;
    }
};

encoder::encoder() : impl(std::make_unique<implementation>()) {}
encoder::~encoder() = default;
const metadata & encoder::params() const { return impl->meta; }
const std::vector<float> & encoder::router_bias(int layer) const {
    require(layer >= 0 && layer < int(impl->biases.size()), "router layer is out of range");
    return impl->biases[layer];
}

std::shared_ptr<encoder> encoder::load(const std::string & path, const std::string & backend_name,
                                      int device, int n_threads) {
    auto result = std::shared_ptr<encoder>(new encoder());
    auto & state = *result->impl;
    auto & m = state.meta;
    file_context source;
    source.file = gguf_init_from_file(path.c_str(), {true, &source.tensors});
    require(source.file != nullptr, "cannot open companion " + path);
    auto * file = source.file;
    const int64_t architecture = gguf_find_key(file, "general.architecture");
    require(architecture >= 0 && gguf_get_kv_type(file, architecture) == GGUF_TYPE_STRING &&
        std::string(gguf_get_val_str(file, architecture)) == "deepseek41_vision", "invalid companion architecture");
    const std::string vp = "deepseek41.vision.";
    m.layers = read_int(file, vp + "num_hidden_layers"); m.dim = read_int(file, vp + "hidden_size");
    m.heads = read_int(file, vp + "num_attention_heads"); m.intermediate = read_int(file, vp + "intermediate_size");
    m.patch_size = read_int(file, vp + "patch_size"); m.downsample_ratio = read_int(file, vp + "downsample_ratio");
    m.max_image_tokens = read_int(file, vp + "max_image_tokens"); m.min_pixels = read_int(file, vp + "min_pixels");
    m.max_wh_ratio = read_int(file, vp + "max_wh_ratio");
    const int64_t theta = gguf_find_key(file, (vp + "rope_theta").c_str());
    require(theta >= 0 && gguf_get_kv_type(file, theta) == GGUF_TYPE_FLOAT32, "missing or invalid vision rope theta");
    m.rope_theta = gguf_get_val_f32(file, theta);
    m.text_dim = read_int(file, "deepseek41.hidden_size"); m.text_layers = read_int(file, "deepseek41.num_hidden_layers");
    m.image_token_id = read_int(file, "deepseek41.image_token_id");
    const int64_t fingerprint = gguf_find_key(file, "deepseek41.tokenizer_hash");
    require(fingerprint >= 0 && gguf_get_kv_type(file, fingerprint) == GGUF_TYPE_UINT64, "missing or invalid parent tokenizer fingerprint");
    m.tokenizer_hash = gguf_get_val_u64(file, fingerprint);
    require(m.layers > 0 && m.layers <= 128 && m.dim > 0 && m.dim <= 8192 && m.heads > 0 &&
        m.dim % m.heads == 0 && (m.dim / m.heads) % 4 == 0 && m.intermediate > 0 && m.intermediate <= 65536 &&
        m.patch_size > 0 && m.patch_size <= 64 && m.downsample_ratio > 0 && m.downsample_ratio <= 8 &&
        m.max_image_tokens >= 4 && m.max_image_tokens <= 8192 && m.rope_theta > 0 && std::isfinite(m.rope_theta) &&
        m.text_dim > 0 && m.text_dim <= 65536 && m.text_layers > 0 && m.text_layers <= 128,
        "invalid companion geometry");
    std::ifstream input(path, std::ios::binary);
    require(bool(input), "cannot read companion");
    state.start = read_vector(input, file, source.tensors, "image_start");
    state.end = read_vector(input, file, source.tensors, "image_end");
    state.newline = read_vector(input, file, source.tensors, "image_newline");
    require(state.start.size() == size_t(m.text_dim) && state.end.size() == size_t(m.text_dim) && state.newline.size() == size_t(m.text_dim), "delimiter dimensions do not match text model");
    for (int layer = 0; layer < m.text_layers; ++layer) {
        state.biases.push_back(read_vector(input, file, source.tensors, "layers." + std::to_string(layer) + ".ffn.gate.bias_vl"));
        if (layer == 0) m.n_experts = int(state.biases.back().size());
        require(m.n_experts > 0 && m.n_experts <= 4096 && state.biases.back().size() == size_t(m.n_experts), "inconsistent router bias dimensions");
    }
    require(device >= 0, "negative backend device index");
    std::string name = backend_name;
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::toupper(c)); });
    if (name.rfind("GGML_", 0) == 0) name = name.substr(5);
    if (name == "CPU") {
        require(device == 0, "CPU encoder requires device zero");
        state.backend = ggml_backend_cpu_init();
        if (state.backend) ggml_backend_cpu_set_n_threads(state.backend, n_threads > 0 ? n_threads : 4);
    } else {
        int matched = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            auto reg = ggml_backend_dev_backend_reg(dev);
            std::string registered = reg ? ggml_backend_reg_name(reg) : "";
            std::transform(registered.begin(), registered.end(), registered.begin(), [](unsigned char c) { return char(std::toupper(c)); });
            if (!name.empty() && name != registered) continue;
            if (matched++ == device) { state.backend = ggml_backend_dev_init(dev, nullptr); break; }
        }
    }
    require(state.backend != nullptr, "requested backend device is unavailable");
#if defined(TSG_GGML_USE_CUDA)
    if (auto * wrapped = tsg_dsv4_fused_backend_init(state.backend)) {
        state.cuda_backend = state.backend;
        state.backend = wrapped;
    }
#endif
    state.weights_ctx = ggml_init({size_t(gguf_get_n_tensors(file) + 1) * ggml_tensor_overhead() + 4096, nullptr, true});
    require(state.weights_ctx != nullptr, "cannot allocate weight metadata");
    for (int64_t i = 0; i < gguf_get_n_tensors(file); ++i) {
        const std::string tensor_name = gguf_get_tensor_name(file, i);
        if (tensor_name.rfind("vision.", 0) != 0 && tensor_name.rfind("aligner.", 0) != 0) continue;
        auto * original = ggml_get_tensor(source.tensors, tensor_name.c_str());
        require(original && (original->type == GGML_TYPE_BF16 || original->type == GGML_TYPE_F32), "vision matrix must be BF16 or F32: " + tensor_name);
        auto * tensor = ggml_dup_tensor(state.weights_ctx, original);
        ggml_set_name(tensor, tensor_name.c_str());
        state.weights.emplace(tensor_name, tensor);
    }
    state.validate_weights();
    auto * patch_weight = state.weight("vision.patch_embed.proj.weight", 3 * m.patch_size * m.patch_size, m.dim);
    m.bf16_activations = patch_weight->type == GGML_TYPE_BF16;
    state.weights_buffer = ggml_backend_alloc_ctx_tensors(state.weights_ctx, state.backend);
    require(state.weights_buffer != nullptr, "cannot allocate vision weights on selected device");
    ggml_backend_buffer_set_usage(state.weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<char> staging(16 * 1024 * 1024);
    for (const auto & [tensor_name, tensor] : state.weights) {
        const auto index = gguf_find_tensor(file, tensor_name.c_str());
        input.seekg(gguf_get_data_offset(file) + gguf_get_tensor_offset(file, index));
        for (size_t offset = 0; offset < ggml_nbytes(tensor);) {
            const size_t bytes = std::min(staging.size(), ggml_nbytes(tensor) - offset);
            require(bool(input.read(staging.data(), bytes)), "truncated tensor " + tensor_name);
            ggml_backend_tensor_set(tensor, staging.data(), offset, bytes);
            offset += bytes;
        }
    }
    return result;
}

int encoder::span_rows(int height, int width) const {
    const auto & m = impl->meta;
    require(height > 0 && width > 0 && height <= 65536 && width <= 65536, "invalid patch grid");
    const int64_t rows = (height + m.downsample_ratio - 1) / m.downsample_ratio;
    const int64_t cols = (width + m.downsample_ratio - 1) / m.downsample_ratio;
    const int64_t span = rows * (cols + 1) + 2;
    require(span <= m.max_image_tokens && int64_t(height) * width <= INT32_MAX, "image exceeds companion token budget");
    return int(span);
}

int encoder::encode(const float * patches, int height, int width, float * output, int capacity) {
    const int count = span_rows(height, width);
    require(patches && output && capacity >= int64_t(count) * impl->meta.text_dim, "invalid image buffers/capacity");
    auto & state = *impl;
    const auto & m = state.meta;
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.cached || state.cached->height != height || state.cached->width != width ||
        state.cached->tracing != (std::getenv("TS_DSV41_VISION_TRACE_DIR") != nullptr)) {
        state.cached.reset(); // release old scratch before allocating a different grid
        state.cached = state.build(height, width);
    }
    auto & g = *state.cached;
    const int n = height * width, half = m.dim / m.heads / 2;
    const int ratio = m.downsample_ratio, mh = (height + ratio - 1) / ratio, mw = (width + ratio - 1) / ratio;
    std::vector<float> cos(size_t(n) * half), sin(cos.size());
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) for (int i = 0; i < half; ++i) {
        const float exponent = float(2 * (i % (half / 2))) / half;
        // Preserve the official F32 inverse-frequency rounding before the
        // position multiplication; division reassociation can cross BF16 bins.
        const float inv_frequency = 1.0f / std::pow(m.rope_theta, exponent);
        const float angle = float(i < half / 2 ? y : x) * inv_frequency;
        cos[size_t(y * width + x) * half + i] = std::cos(angle);
        sin[size_t(y * width + x) * half + i] = std::sin(angle);
    }
    std::vector<int32_t> ids(size_t(mh) * mw * ratio * ratio);
    for (int by = 0; by < mh; ++by) for (int bx = 0; bx < mw; ++bx)
        for (int dy = 0; dy < ratio; ++dy) for (int dx = 0; dx < ratio; ++dx) {
            const int y = by * ratio + dy, x = bx * ratio + dx;
            ids[(size_t(by * mw + bx) * ratio + dy) * ratio + dx] = y < height && x < width ? y * width + x : n;
        }
    const std::vector<float> zeros(m.dim, 0.0f);
    ggml_backend_tensor_set(g.patches, patches, 0, ggml_nbytes(g.patches));
    ggml_backend_tensor_set(g.cos, cos.data(), 0, cos.size() * sizeof(float));
    ggml_backend_tensor_set(g.sin, sin.data(), 0, sin.size() * sizeof(float));
    ggml_backend_tensor_set(g.zero, zeros.data(), 0, zeros.size() * sizeof(float));
    ggml_backend_tensor_set(g.gather, ids.data(), 0, ids.size() * sizeof(int32_t));
    require(ggml_backend_graph_compute(state.backend, g.gf) == GGML_STATUS_SUCCESS, "vision graph computation failed");
    if (g.tracing) {
        const std::filesystem::path directory(std::getenv("TS_DSV41_VISION_TRACE_DIR"));
        std::filesystem::create_directories(directory);
        for (const auto & [name, tensor] : g.traces) {
            std::vector<float> values(ggml_nelements(tensor));
            ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
            std::ofstream file(directory / (name + ".f32"), std::ios::binary);
            require(bool(file.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float))), "cannot write vision trace " + name);
        }
    }
    std::vector<float> aligned(size_t(mh) * mw * m.text_dim);
    ggml_backend_tensor_get(g.output, aligned.data(), 0, aligned.size() * sizeof(float));
    for (float value : aligned) require(std::isfinite(value), "vision produced nonfinite embeddings");
    int row = 0;
    auto copy = [&](const float * src, int rows) {
        std::memcpy(output + size_t(row) * m.text_dim, src, size_t(rows) * m.text_dim * sizeof(float));
        row += rows;
    };
    copy(state.start.data(), 1);
    for (int y = 0; y < mh; ++y) { copy(aligned.data() + size_t(y) * mw * m.text_dim, mw); copy(state.newline.data(), 1); }
    copy(state.end.data(), 1);
    require(row == count, "internal image-span size mismatch");
    return count;
}

struct handle { std::shared_ptr<encoder> value; };
std::shared_ptr<encoder> from_handle(void * opaque) {
    require(opaque != nullptr, "null encoder handle");
    return static_cast<handle *>(opaque)->value;
}
} // namespace tsg_dsv41_vision

TSG_EXPORT void * TSGgml_Dsv41VisionLoad(const char * path, const char * backend, int device, int threads) {
    try {
        if (!path) throw std::runtime_error("DeepSeek V4.1 vision: null companion path");
        return new tsg_dsv41_vision::handle{tsg_dsv41_vision::encoder::load(path, backend ? backend : "CUDA", device, threads)};
    } catch (const std::exception & error) { tsg::set_last_error(error.what()); return nullptr; }
}
TSG_EXPORT void TSGgml_Dsv41VisionFree(void * handle) { delete static_cast<tsg_dsv41_vision::handle *>(handle); }
TSG_EXPORT int TSGgml_Dsv41VisionInfo(void * handle, int32_t * info, int count) {
    try {
        if (!info || count < 8) throw std::runtime_error("DeepSeek V4.1 vision: info requires eight integers");
        const auto & m = tsg_dsv41_vision::from_handle(handle)->params();
        const int32_t values[] = {m.patch_size, m.downsample_ratio, m.text_dim, m.max_image_tokens,
                                 m.min_pixels, m.max_wh_ratio, m.image_token_id, m.dim};
        std::copy(std::begin(values), std::end(values), info);
        return 0;
    } catch (const std::exception & error) { tsg::set_last_error(error.what()); return -1; }
}
TSG_EXPORT int TSGgml_Dsv41VisionEncode(void * handle, const float * patches, int height, int width,
                                      float * output, int capacity) {
    try { return tsg_dsv41_vision::from_handle(handle)->encode(patches, height, width, output, capacity); }
    catch (const std::exception & error) { tsg::set_last_error(error.what()); return -1; }
}
