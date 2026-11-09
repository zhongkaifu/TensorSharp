// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_internal.h"
#include <cstdlib>
#include <cstddef>
#include <memory>
#include "ggml_ops_matmul_precision.h"
#include "embedding_audio_cpu_projection.h"
#include "ggml-cpu.h"
#if defined(TSG_GGML_USE_CUDA)
#include "ggml_ops_dsv4_fused.h"
#endif
#include <cstring>
#if defined(TSG_GGML_USE_METAL)
#include "ggml_gemma4_audio_metal_precision.h"
#endif

using namespace tsg;

namespace {

bool precise_rows_enabled();
bool mps_projection_enabled(bool require_float32_precision = false);
bool fused_ops_enabled(bool qualified_default = false);
bool cuda_pedantic_enabled(bool require_float32_precision = false);
bool cuda_f32_products_enabled();
bool dense_projection_enabled(bool require_float32_precision);

struct audio_graph {
    // Declare before context so the provider outlives every callback's graph
    // metadata, including a context returned to the owned pool on destruction.
    tsg_embeddings::audio_cpu_projection cpu_projection;
    PooledContextHandle context;
    std::vector<BufferHandle> host_buffers;
    struct binding {
        TensorView2DDesc desc;
        TensorBinding tensor;
        bool mapped;
    };
    std::vector<binding> bindings;
    struct immutable_upload { ggml_tensor * tensor; void * data; size_t bytes; };
    std::vector<immutable_upload> immutable_uploads;

    int graph_capacity;
    bool require_float32_precision;
    bool use_fused_operations;

    explicit audio_graph(int blocks = 1, int projection_rows = 0, int projection_linears = 10, bool require_precision = false)
        : cpu_projection(g_backend_type == BACKEND_TYPE_CPU, require_precision,
              std::getenv("TSG_EMBEDDING_LINUX_BLAS"), std::getenv("TSG_EMBEDDING_BLAS_LIBRARY"),
              ggml_cpu_has_llamafile() && ggml_cpu_has_avx()),
          graph_capacity(std::max(2048, blocks * 512)), require_float32_precision(require_precision),
          use_fused_operations(fused_ops_enabled()) {
        size_t context_bytes = static_cast<size_t>(2 * 1024 * 1024) * blocks;
        if (precise_rows_enabled() && projection_rows > 0) {
            // Each projected row adds one view, one matvec, and at most one
            // concat. The 512-node allowance covers weights and all remaining
            // block operations. Reserve both graph slots and tensor metadata.
            const int64_t nodes = static_cast<int64_t>(blocks) * (512LL + 3LL * projection_rows * projection_linears);
            constexpr int64_t maximum_nodes = 1LL << 20;
            constexpr size_t maximum_context_bytes = 512ULL * 1024 * 1024;
            if (nodes > maximum_nodes)
                throw std::runtime_error("Gemma4 audio precise-row diagnostic exceeds its bounded graph capacity.");
            graph_capacity = std::max(graph_capacity, static_cast<int>(nodes));
            const size_t metadata = static_cast<size_t>(graph_capacity) * (ggml_tensor_overhead() + 64) +
                ggml_graph_overhead_custom(graph_capacity, false) + 8192;
            context_bytes = std::max(context_bytes, metadata);
            if (context_bytes > maximum_context_bytes)
                throw std::runtime_error("Gemma4 audio precise-row diagnostic exceeds its bounded context memory.");
        }
        if (!context.init(context_bytes)) throw std::runtime_error("Cannot allocate audio graph context.");
        bindings.reserve(6);
    }

    ggml_tensor * bind(const TensorView2DDesc& desc) {
        TensorBinding tensor;
        ggml_backend_buffer_t buffer = nullptr;
        const bool mapped = create_binding_from_host_ptr_2d(context.value, g_backend, desc, tensor, buffer);
        if (mapped) host_buffers.emplace_back(buffer);
        else tensor = create_standard_binding(context.value, desc);
        if (!tensor.tensor || !tensor.storage) throw std::runtime_error("Cannot bind audio graph tensor.");
        bindings.push_back({ desc, tensor, mapped });
        return tensor.tensor;
    }

    ggml_tensor * bind_immutable(const TensorView2DDesc& desc) {
        if (desc.stride1 != 1 || desc.stride0 != desc.dim1) return bind(desc);
        auto * tensor = ggml_new_tensor_2d(context.value, GGML_TYPE_F32, desc.dim1, desc.dim0);
        bool upload = false;
        if (try_bind_cached_tensor(g_backend, ggml_backend_get_device(g_backend), tensor, desc.data,
            static_cast<size_t>(desc.raw_bytes), upload, GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) {
            if (upload) immutable_uploads.push_back({tensor, desc.data, static_cast<size_t>(desc.raw_bytes)});
            return tensor;
        }
        return bind(desc);
    }

    ggml_tensor * bind_quantized(void * data, ggml_type type, int input, int output, size_t bytes) {
        auto * tensor = ggml_new_tensor_2d(context.value, type, input, output);
        bool upload = false;
        if (!try_bind_cached_tensor(g_backend, ggml_backend_get_device(g_backend), tensor, data,
            bytes, upload, GGML_BACKEND_BUFFER_USAGE_WEIGHTS) || upload)
            immutable_uploads.push_back({tensor, data, bytes});
        return tensor;
    }

    int execute(ggml_tensor * value, const std::vector<std::pair<ggml_tensor *, std::vector<float>>>& constants) {
        const auto& result = bindings[0];
        auto * output = ggml_cpy(context.value, value, result.tensor.tensor);
        ggml_set_output(output);
        auto * graph = ggml_new_graph_custom(context.value, graph_capacity, false);
        ggml_build_forward_expand(graph, output);
        BufferHandle buffer(nullptr);
        if (!alloc_ctx_tensors_reuse(context.value, graph)) {
            buffer = BufferHandle(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
            if (!buffer.value) throw std::runtime_error("Cannot allocate audio graph backend buffer.");
        }
        // Preserve any gaps in a strided output view. Inputs and outputs are
        // bound independently; unified-memory mappings do not require uploads.
        for (const auto& item : bindings)
            if (!item.mapped) upload_binding(item.tensor, item.desc.data, item.tensor.raw_bytes);
        for (const auto& item : constants)
            ggml_backend_tensor_set(item.first, item.second.data(), 0, item.second.size() * sizeof(float));
        for (const auto& item : immutable_uploads)
            ggml_backend_tensor_set(item.tensor, item.data, 0, item.bytes);
        ggml_status status = GGML_STATUS_SUCCESS;
        bool handled = false;
#if defined(TSG_GGML_USE_METAL)
        const auto * precise_mm = std::getenv("TS_GEMMA4_AUDIO_METAL_F32_MM");
        if (g_backend_type == BACKEND_TYPE_METAL &&
            ((precise_mm && std::strcmp(precise_mm, "1") == 0) || mps_projection_enabled(require_float32_precision) || fused_ops_enabled()))
            handled = tsg_gemma4_audio_metal_graph_compute(g_backend, graph, &status, require_float32_precision);
#endif
#if defined(TSG_GGML_USE_CUDA)
        if (cuda_pedantic_enabled(require_float32_precision) || cuda_f32_products_enabled()) {
            // A scoped owned backend dispatches custom F32 GEMMs with its
            // private pedantic cuBLAS handle. Ordinary ggml CUDA nodes retain
            // the original stream; no shared handle math mode is mutated.
            std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> precise(
                tsg_dsv4_fused_backend_try_init(g_backend), &ggml_backend_free);
            if (!precise) throw std::runtime_error("Cannot create scoped precise CUDA audio backend.");
            status = ggml_backend_graph_compute(precise.get(), graph);
            handled = true;
        }
#endif
        if (!handled && require_float32_precision && g_backend_type == BACKEND_TYPE_METAL)
            throw std::runtime_error("Required F32 Metal audio execution rejected this graph; ordinary narrowing is disabled.");
        if (!handled) status = compute_graph(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Audio graph execution failed on the selected ggml backend.");
        const char * profile = std::getenv("TSG_EMBEDDING_CPU_PROFILE");
        if (g_backend_type == BACKEND_TYPE_CPU && profile && std::strcmp(profile, "1") == 0) {
            const auto counts = cpu_projection.counts();
            std::fprintf(stderr, "TS_AUDIO_CPU_BLAS available=%d nodes=%zu gemm_calls=%llu worker_mask=%llu output_elements=%llu shape_rejections=%llu policy_rejections=%llu active_calls=%llu\n",
                cpu_projection.available(), cpu_projection.nodes(), (unsigned long long)counts.gemm_calls,
                (unsigned long long)counts.worker_mask, (unsigned long long)counts.output_elements,
                (unsigned long long)counts.shape_rejections, (unsigned long long)counts.policy_rejections,
                (unsigned long long)counts.active_calls);
        }
        finalize_compute(result.mapped, result.tensor.storage, result.desc.data, result.tensor.raw_bytes);
        clear_last_error();
        return 1;
    }
};

bool matrix(const TensorView2DDesc& desc, int rows, int columns, const char * name) {
    if (!validate_desc(desc, name)) return false;
    if (desc.dim0 != rows || desc.dim1 != columns || !can_map_standard_view(desc)) {
        set_last_error("Gemma4 audio requires row-contiguous F32 matrices of the declared shape.");
        return false;
    }
    return true;
}

ggml_tensor * pad_time(audio_graph& graph, ggml_tensor * input, int channels, int past, int future,
    std::vector<std::pair<ggml_tensor *, std::vector<float>>>& constants) {
    auto * ctx = graph.context.value;
    if (past) {
        auto * padding = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels, past);
        constants.emplace_back(padding, std::vector<float>(static_cast<size_t>(channels) * past));
        input = ggml_concat(ctx, padding, input, 1);
    }
    if (future) {
        auto * padding = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels, future);
        constants.emplace_back(padding, std::vector<float>(static_cast<size_t>(channels) * future));
        input = ggml_concat(ctx, input, padding, 1);
    }
    return ggml_cont(ctx, input);
}

ggml_tensor * full_precision_mul(ggml_context * ctx, ggml_tensor * left, ggml_tensor * right, bool require_float32_precision = false) {
    if (cuda_pedantic_enabled(require_float32_precision) || (cuda_f32_products_enabled() && !ggml_is_quantized(left->type)))
        return tsg_matmul_f32(ctx, left, right);
    auto * result = ggml_mul_mat(ctx, left, right);
    ggml_prec_set_acc(result, GGML_PREC_F32);
    return result;
}

// A diagnostic for Metal's matrix kernels, which stage F32 operands in F16.
// One-row views keep the inner dimension contiguous and select the upstream
// F32 matvec path. It is intentionally opt-in because of the dispatch cost.
bool mps_projection_enabled(bool require_float32_precision) {
#if defined(TSG_GGML_USE_METAL)
    const auto * value = std::getenv("TS_GEMMA4_AUDIO_METAL_MPS_MM");
    return g_backend_type == BACKEND_TYPE_METAL && !(value && std::strcmp(value, "0") == 0) &&
        (require_float32_precision || (value && std::strcmp(value, "1") == 0));
#else
    (void)require_float32_precision;
    return false;
#endif
}

bool cuda_pedantic_enabled(bool require_float32_precision) {
#if defined(TSG_GGML_USE_CUDA)
    const auto * value = std::getenv("TS_GEMMA4_AUDIO_GGML_CUDA_PEDANTIC");
    return g_backend_type == BACKEND_TYPE_CUDA && !(value && std::strcmp(value, "0") == 0) &&
        (require_float32_precision || (value && std::strcmp(value, "1") == 0));
#else
    (void)require_float32_precision;
    return false;
#endif
}

// ggml-cuda creates every cuBLAS handle with CUBLAS_TF32_TENSOR_OP_MATH (common.cuh), so an F32
// ggml_mul_mat that reaches cuBLAS (a projection of more than three rows; one to three rows take
// ggml's exact F32 vector kernel) may run on TF32 tensor cores although full_precision_mul asks
// for GGML_PREC_F32: that selects the F32 compute type, not the math mode. On the 5090 the
// conformer fixture's 25- and 96-row blocks came out 5.8e-4 / 7.1e-4 from the double reference,
// against 4.3e-7 / 7.6e-7 with these products on the pedantic handle. So F32 (and F16/BF16)
// products of these graphs go through TensorSharp's pedantic F32 GEMM on CUDA by default;
// TS_GEMMA4_AUDIO_GGML_CUDA_PEDANTIC=0 keeps ggml's TF32 cuBLAS products. Quantized (Q8_0)
// projections keep ggml's integer kernels unless pedantic dense projections are asked for.
bool cuda_f32_products_enabled() {
#if defined(TSG_GGML_USE_CUDA)
    const auto * value = std::getenv("TS_GEMMA4_AUDIO_GGML_CUDA_PEDANTIC");
    return g_backend_type == BACKEND_TYPE_CUDA && !(value && std::strcmp(value, "0") == 0);
#else
    return false;
#endif
}

bool dense_projection_enabled(bool require_float32_precision) {
    return mps_projection_enabled(require_float32_precision) || cuda_pedantic_enabled(require_float32_precision);
}

bool fused_ops_enabled(bool qualified_default) {
#if defined(TSG_GGML_USE_METAL)
    const auto * value = std::getenv("TS_GEMMA4_AUDIO_METAL_FUSED_OPS");
    return g_backend_type == BACKEND_TYPE_METAL &&
        (value ? std::strcmp(value, "1") == 0 : qualified_default);
#else
    (void)qualified_default;
    return false;
#endif
}

bool precise_rows_enabled() {
    const auto * value = std::getenv("TS_GEMMA4_AUDIO_PRECISE_ROWS");
    return g_backend_type == BACKEND_TYPE_METAL && value && std::strcmp(value, "1") == 0;
}

ggml_tensor * projection_mul(audio_graph& graph, ggml_tensor * left, ggml_tensor * right) {
    auto * ctx = graph.context.value;
    if (auto * output = graph.cpu_projection.multiply(ctx, left, right)) return output;
    const bool require_float32_precision = graph.require_float32_precision;
    if (!precise_rows_enabled() || right->ne[1] <= 1 || right->ne[2] != 1 || right->ne[3] != 1) {
        auto * result = full_precision_mul(ctx, left, right, require_float32_precision);
        if (right->ne[1] > 1 && right->ne[2] == 1 && right->ne[3] == 1)
            ggml_set_name(result, "tsg_gemma4_audio_f32_projection");
        return result;
    }
    ggml_tensor * output = nullptr;
    for (int64_t row = 0; row < right->ne[1]; ++row) {
        auto * input = ggml_view_2d(ctx, right, right->ne[0], 1, right->nb[1], row * right->nb[1]);
        auto * current = full_precision_mul(ctx, left, input, require_float32_precision);
        output = output ? ggml_concat(ctx, output, current, 1) : current;
    }
    return output;
}

ggml_tensor * window_view(ggml_context * ctx, ggml_tensor * source,
    int64_t first, int64_t second, int64_t third, int64_t fourth,
    size_t nb1, size_t nb2, size_t nb3, size_t nb0 = sizeof(float)) {
    // ggml_view_4d initially checks the dense logical volume before applying
    // strides. An overlapping window legitimately has a larger logical volume
    // than its storage. Create a bounded 1D view, then describe the owned view
    // with its actual strides; do not enlarge or modify the source allocation.
    auto * result = ggml_view_1d(ctx, source, ggml_nelements(source), 0);
    result->ne[0] = first; result->ne[1] = second;
    result->ne[2] = third; result->ne[3] = fourth;
    result->nb[0] = nb0; result->nb[1] = nb1;
    result->nb[2] = nb2; result->nb[3] = nb3;
    if (ggml_nbytes(result) > ggml_nbytes(source))
        throw std::runtime_error("Gemma4 audio window exceeds its padded source storage.");
    return result;
}

ggml_tensor * build_attention(audio_graph& graph, ggml_tensor * q, ggml_tensor * k,
    ggml_tensor * v, ggml_tensor * r, int sequence, int heads, int head,
    int past, int future, float cap,
    std::vector<std::pair<ggml_tensor *, std::vector<float>>>& constants) {
    auto * ctx = graph.context.value;
    const int hidden = heads * head, window = past + future + 1;
    q = ggml_cont(ctx, q);
    r = ggml_cont(ctx, r);
#if defined(TSG_GGML_USE_METAL)
    if (graph.use_fused_operations)
        if (auto * output = tsg_gemma4_audio_metal_attention(ctx, q, k, v, r,
            sequence, heads, head, past, future, cap)) return output;
#endif
    k = pad_time(graph, k, hidden, past, future, constants);
    v = pad_time(graph, v, hidden, past, future, constants);

    const size_t channel_bytes = static_cast<size_t>(head) * sizeof(float);
    const size_t row_bytes = static_cast<size_t>(hidden) * sizeof(float);
    auto * queries = ggml_reshape_4d(ctx, q, head, 1, heads, sequence);
    // Overlapping views gather only the permitted window, retaining linear
    // rather than quadratic score storage. Heads remain independent batches.
    auto * keys = window_view(ctx, k, head, window, heads, sequence,
        row_bytes, channel_bytes, row_bytes);
    keys = ggml_cont(ctx, keys);
    auto * positions = ggml_reshape_3d(ctx, r, head, heads, window);
    positions = ggml_cont(ctx, ggml_permute(ctx, positions, 0, 2, 1, 3));
    auto * scores = ggml_add(ctx, full_precision_mul(ctx, keys, queries, graph.require_float32_precision),
        full_precision_mul(ctx, positions, queries, graph.require_float32_precision));
    scores = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, scores, 1.f / cap)), cap);
    auto * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, window, 1, 1, sequence);
    std::vector<float> mask_values(static_cast<size_t>(window) * sequence);
    for (int time = 0; time < sequence; ++time)
        for (int index = 0; index < window; ++index) {
            const int64_t key_time = static_cast<int64_t>(time) + index - past;
            if (key_time < 0 || key_time >= sequence)
                mask_values[static_cast<size_t>(time) * window + index] = -1.e9f;
        }
    constants.emplace_back(mask, std::move(mask_values));
    // Apply padding after the soft cap, matching the independent CPU oracle.
    scores = ggml_soft_max(ctx, ggml_add(ctx, scores, mask));
    auto * values = window_view(ctx, v, window, head, heads, sequence,
        sizeof(float), channel_bytes, row_bytes, row_bytes);
    values = ggml_cont(ctx, values);
    auto * output = full_precision_mul(ctx, values, scores, graph.require_float32_precision);
    output = ggml_cont(ctx, ggml_permute(ctx, output, 0, 3, 1, 2));
    output = ggml_reshape_2d(ctx, output, hidden, sequence);
    return output;
}

int attention(const TensorView2DDesc& result, const TensorView2DDesc& query,
    const TensorView2DDesc& key, const TensorView2DDesc& value, const TensorView2DDesc& relative,
    int sequence, int heads, int head, int past, int future, float cap) {
    const int64_t hidden64 = static_cast<int64_t>(heads) * head;
    const int64_t window64 = static_cast<int64_t>(past) + future + 1;
    if (sequence <= 0 || heads <= 0 || head <= 0 || past < 0 || future < 0 ||
        hidden64 > std::numeric_limits<int>::max() || window64 > std::numeric_limits<int>::max() ||
        !std::isfinite(cap) || cap <= 0) {
        set_last_error("Invalid Gemma4 audio attention geometry or soft cap."); return 0;
    }
    const int hidden = static_cast<int>(hidden64), window = static_cast<int>(window64);
    if (!ensure_backend() || !matrix(result, sequence, hidden, "result") ||
        !matrix(query, sequence, hidden, "query") || !matrix(key, sequence, hidden, "key") ||
        !matrix(value, sequence, hidden, "value") || !matrix(relative, window, hidden, "relative")) return 0;
    if (g_backend_type == BACKEND_TYPE_METAL && sequence > INT16_MAX) {
        set_last_error("Gemma4 audio attention sequence exceeds the Metal batch geometry limit."); return 0;
    }

    audio_graph graph;
    auto * ctx = graph.context.value;
    graph.bind(result);
    auto * q = ggml_cont(ctx, graph.bind(query));
    auto * k = graph.bind(key);
    auto * v = graph.bind(value);
    auto * r = ggml_cont(ctx, graph.bind(relative));
    std::vector<std::pair<ggml_tensor *, std::vector<float>>> constants;
    auto * output = build_attention(graph, q, k, v, r, sequence, heads, head, past, future, cap, constants);
    return graph.execute(output, constants);
}

ggml_tensor * build_depthwise(audio_graph& graph, ggml_tensor * x, ggml_tensor * w,
    int sequence, int channels, int kernel,
    std::vector<std::pair<ggml_tensor *, std::vector<float>>>& constants) {
    auto * ctx = graph.context.value;
    w = ggml_cont(ctx, w);
#if defined(TSG_GGML_USE_METAL)
    if (graph.use_fused_operations)
        if (auto * output = tsg_gemma4_audio_metal_depthwise(ctx, x, w,
            sequence, channels, kernel)) return output;
#endif
    x = pad_time(graph, x, channels, kernel - 1, 0, constants);
    const size_t row_bytes = static_cast<size_t>(channels) * sizeof(float);
    auto * windows = window_view(ctx, x, kernel, channels, 1, sequence,
        sizeof(float), row_bytes * kernel, row_bytes, row_bytes);
    windows = ggml_cont(ctx, windows);
    windows = ggml_reshape_4d(ctx, windows, kernel, 1, channels, sequence);
    w = ggml_reshape_4d(ctx, w, kernel, 1, channels, 1);
    auto * output = full_precision_mul(ctx, w, windows, graph.require_float32_precision);
    output = ggml_cont(ctx, ggml_permute(ctx, output, 2, 3, 0, 1));
    output = ggml_reshape_2d(ctx, output, channels, sequence);
    return output;
}

int depthwise(const TensorView2DDesc& result, const TensorView2DDesc& input,
    const TensorView2DDesc& weight, int sequence, int channels, int kernel) {
    if (sequence <= 0 || channels <= 0 || kernel <= 0) {
        set_last_error("Invalid Gemma4 audio depthwise geometry."); return 0;
    }
    if (!ensure_backend() || !matrix(result, sequence, channels, "result") ||
        !matrix(input, sequence, channels, "input") || !matrix(weight, channels, kernel, "weight")) return 0;
    if (g_backend_type == BACKEND_TYPE_METAL && sequence > INT16_MAX) {
        set_last_error("Gemma4 audio depthwise sequence exceeds the Metal batch geometry limit."); return 0;
    }
    audio_graph graph;
    auto * ctx = graph.context.value;
    graph.bind(result);
    auto * x = graph.bind(input);
    auto * w = ggml_cont(ctx, graph.bind(weight));
    std::vector<std::pair<ggml_tensor *, std::vector<float>>> constants;
    auto * output = build_depthwise(graph, x, w, sequence, channels, kernel, constants);
    return graph.execute(output, constants);
}

struct audio_linear_desc {
    TensorView2DDesc weight, bias;
    int has_clamp;
    float in_min, in_max, out_min, out_max;
    void * compressed_weight;
    int compressed_type;
    int64_t compressed_bytes;
};

struct audio_block_args {
    int struct_bytes;
    TensorView2DDesc result, input, relative;
    const TensorView2DDesc * weights;
    const audio_linear_desc * linears;
    int num_weights, num_linears, linear_struct_bytes;
    int sequence, heads, head, past, future, kernel;
    float epsilon, cap, grad_clip, residual_weight;
    int require_float32_precision;
    int precision_policy_version;
};

struct audio_sscp_conv_desc {
    TensorView2DDesc weight, norm_weight, norm_bias;
    int input_channels, output_channels, kernel_frequency, kernel_time;
};

struct audio_sscp_args {
    int struct_bytes;
    TensorView2DDesc result, input;
    const audio_sscp_conv_desc * convolutions;
    const audio_linear_desc * projection;
    int convolution_struct_bytes, linear_struct_bytes;
    int frequency, time, channels;
    float epsilon;
    int require_float32_precision;
};

static_assert(sizeof(void *) != 8 || sizeof(audio_sscp_conv_desc) == 112, "Audio SSCP convolution ABI mismatch.");
static_assert(sizeof(void *) != 8 || sizeof(audio_sscp_args) == 120, "Audio SSCP argument ABI mismatch.");
static_assert(sizeof(void *) != 8 || sizeof(audio_block_args) == 184, "Audio conformer argument ABI mismatch.");
static_assert(sizeof(void *) != 8 || offsetof(audio_block_args, require_float32_precision) == 172, "Audio conformer precision offset mismatch.");
static_assert(sizeof(void *) != 8 || offsetof(audio_block_args, precision_policy_version) == 176, "Audio conformer version offset mismatch.");
static_assert(sizeof(void *) != 8 || offsetof(audio_sscp_args, require_float32_precision) == 112, "Audio SSCP precision offset mismatch.");

int sscp(const audio_sscp_args * args) {
    if (!args || args->struct_bytes != sizeof(audio_sscp_args) || !args->convolutions || !args->projection ||
        args->convolution_struct_bytes != sizeof(audio_sscp_conv_desc) || args->linear_struct_bytes != sizeof(audio_linear_desc) ||
        args->frequency <= 0 || args->time <= 0 || args->channels <= 0 ||
        !std::isfinite(args->epsilon) || args->epsilon <= 0 || args->require_float32_precision < 0 || args->require_float32_precision > 1 ||
        static_cast<int64_t>(args->frequency) * args->channels > std::numeric_limits<int>::max()) {
        set_last_error("Invalid Gemma4 audio SSCP descriptors or geometry."); return 0;
    }
    int frequency = args->frequency, time = args->time, channels = args->channels;
    if (!ensure_backend() || !matrix(args->input, time, frequency * channels, "SSCP input")) return 0;
    for (int layer = 0; layer < 2; ++layer) {
        const auto& conv = args->convolutions[layer];
        const int64_t columns = static_cast<int64_t>(conv.input_channels) * conv.kernel_frequency * conv.kernel_time;
        if (conv.input_channels != channels || conv.output_channels <= 0 || conv.kernel_frequency <= 0 || conv.kernel_time <= 0 ||
            columns > std::numeric_limits<int>::max() || !matrix(conv.weight, conv.output_channels, static_cast<int>(columns), "SSCP kernel") ||
            (conv.norm_weight.data && !matrix(conv.norm_weight, 1, conv.output_channels, "SSCP norm weight")) ||
            (conv.norm_bias.data && !matrix(conv.norm_bias, 1, conv.output_channels, "SSCP norm bias"))) return 0;
        const int64_t output_frequency = (static_cast<int64_t>(frequency) + 2 - conv.kernel_frequency) / 2 + 1;
        const int64_t output_time = (static_cast<int64_t>(time) + 2 - conv.kernel_time) / 2 + 1;
        if (frequency + 2LL < conv.kernel_frequency || time + 2LL < conv.kernel_time || output_frequency <= 0 || output_time <= 0 ||
            output_frequency * output_time > std::numeric_limits<int>::max() || output_frequency * conv.output_channels > std::numeric_limits<int>::max()) {
            set_last_error("Gemma4 audio SSCP convolution has invalid output dimensions."); return 0;
        }
        frequency = static_cast<int>(output_frequency); time = static_cast<int>(output_time); channels = conv.output_channels;
    }
    const auto& projection = *args->projection;
    const int columns = frequency * channels, outputs = projection.weight.dim0;
    if (outputs <= 0 || !matrix(projection.weight, outputs, columns, "SSCP projection") ||
        (projection.bias.data && !matrix(projection.bias, 1, outputs, "SSCP projection bias")) ||
        !matrix(args->result, time, outputs, "SSCP result")) return 0;
    if (projection.compressed_weight && (projection.compressed_type != GGML_TYPE_Q8_0 || columns % 32 != 0 ||
        projection.compressed_bytes != static_cast<int64_t>(ggml_row_size(GGML_TYPE_Q8_0, columns)) * outputs)) {
        set_last_error("Gemma4 audio SSCP compressed projection requires exact Q8_0 blocks."); return 0;
    }
    if (projection.has_clamp && (!std::isfinite(projection.in_min) || !std::isfinite(projection.in_max) ||
        !std::isfinite(projection.out_min) || !std::isfinite(projection.out_max) || projection.in_min > projection.in_max || projection.out_min > projection.out_max)) {
        set_last_error("Invalid Gemma4 audio SSCP projection clamp."); return 0;
    }
    audio_graph graph(1, time, 1, args->require_float32_precision != 0);
    auto * ctx = graph.context.value;
    graph.bind(args->result);
    auto * input = graph.bind(args->input);
    // Public rows are [time,frequency*channels+channel]. Convolution uses
    // [frequency,time,channel], while channel normalization needs ne0=channel.
    auto * x = ggml_reshape_3d(ctx, input, args->channels, args->frequency, args->time);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));
    frequency = args->frequency; time = args->time; channels = args->channels;
    for (int layer = 0; layer < 2; ++layer) {
        const auto& conv = args->convolutions[layer];
        auto * kernel = graph.bind_immutable(conv.weight);
        const int kernel_columns = channels * conv.kernel_frequency * conv.kernel_time;
        kernel = ggml_reshape_4d(ctx, kernel, conv.kernel_frequency, conv.kernel_time, channels, conv.output_channels);
        const int output_frequency = (frequency + 2 - conv.kernel_frequency) / 2 + 1;
        const int output_time = (time + 2 - conv.kernel_time) / 2 + 1;
        if (g_backend_type == BACKEND_TYPE_METAL) {
            // The upstream direct operation reads/accumulates F32; its GEMM
            // alternative stages both F32 operands in F16 on Metal.
            x = ggml_conv_2d_direct(ctx, kernel, x, 2, 2, 1, 1, 1, 1);
            x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
            x = ggml_reshape_2d(ctx, x, conv.output_channels, output_frequency * output_time);
        } else {
            auto * columns_tensor = ggml_im2col(ctx, kernel, x, 2, 2, 1, 1, 1, 1, true, GGML_TYPE_F32);
            columns_tensor = ggml_reshape_2d(ctx, columns_tensor, kernel_columns, output_frequency * output_time);
            x = full_precision_mul(ctx, ggml_reshape_2d(ctx, kernel, kernel_columns, conv.output_channels), columns_tensor, graph.require_float32_precision);
        }
        if (conv.norm_weight.data || conv.norm_bias.data) {
            x = ggml_norm(ctx, x, args->epsilon);
            if (conv.norm_weight.data) x = ggml_mul(ctx, x, graph.bind_immutable(conv.norm_weight));
            if (conv.norm_bias.data) x = ggml_add(ctx, x, graph.bind_immutable(conv.norm_bias));
        }
        x = ggml_relu(ctx, x);
        x = ggml_reshape_3d(ctx, x, conv.output_channels, output_frequency, output_time);
        if (layer == 0) x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));
        frequency = output_frequency; time = output_time; channels = conv.output_channels;
    }
    x = ggml_reshape_2d(ctx, x, columns, time);
    if (projection.has_clamp) x = ggml_clamp(ctx, x, projection.in_min, projection.in_max);
    // The opt-in MPS path consumes the existing exact dequantized F32 tensor.
    // Its independent immutable cache key preserves original Q8 ownership.
    auto * weight = projection.compressed_weight && !dense_projection_enabled(graph.require_float32_precision)
        ? graph.bind_quantized(projection.compressed_weight, GGML_TYPE_Q8_0, columns, outputs, static_cast<size_t>(projection.compressed_bytes))
        : graph.bind_immutable(projection.weight);
    x = projection_mul(graph, weight, x);
    if (projection.bias.data) x = ggml_add(ctx, x, graph.bind_immutable(projection.bias));
    if (projection.has_clamp) x = ggml_clamp(ctx, x, projection.out_min, projection.out_max);
    return graph.execute(x, {});
}

bool validate_block(const audio_block_args * args) {
    if (!args || args->struct_bytes != sizeof(audio_block_args) || !args->weights || !args->linears ||
        args->num_weights != 11 || args->num_linears != 10 || args->linear_struct_bytes != sizeof(audio_linear_desc) || args->sequence <= 0 ||
        args->heads <= 0 || args->head <= 0 || args->past < 0 || args->future < 0 || args->kernel <= 0 ||
        static_cast<int64_t>(args->heads) * args->head > std::numeric_limits<int>::max() ||
        static_cast<int64_t>(args->past) + args->future + 1 > std::numeric_limits<int>::max() ||
        !std::isfinite(args->epsilon) || args->epsilon <= 0 || !std::isfinite(args->cap) || args->cap <= 0 ||
        !std::isfinite(args->grad_clip) || args->grad_clip <= 0 || !std::isfinite(args->residual_weight) ||
        args->require_float32_precision < 0 || args->require_float32_precision > 1 || args->precision_policy_version != 1) {
        set_last_error("Invalid Gemma4 conformer block descriptors or geometry."); return 0;
    }
    const int hidden = args->heads * args->head, sequence = args->sequence;
    if (!ensure_backend() || !matrix(args->result, sequence, hidden, "result") ||
        !matrix(args->input, sequence, hidden, "input") ||
        !matrix(args->relative, args->past + args->future + 1, hidden, "relative")) return 0;
    if (g_backend_type == BACKEND_TYPE_METAL && sequence > INT16_MAX) {
        set_last_error("Gemma4 conformer sequence exceeds the Metal batch geometry limit."); return 0;
    }
    for (int i = 0; i < 11; ++i) {
        const auto& weight = args->weights[i];
        if (!weight.data) continue;
        const int rows = i == 10 ? hidden : 1;
        const int columns = i == 10 ? args->kernel : i == 9 ? args->head : hidden;
        if (!matrix(weight, rows, columns, "conformer weight")) return 0;
    }
    const int ff1 = args->linears[0].weight.dim0, ff2 = args->linears[8].weight.dim0;
    if (ff1 <= 0 || ff2 <= 0 || hidden > std::numeric_limits<int>::max() / 2) {
        set_last_error("Invalid Gemma4 conformer projection dimensions."); return 0;
    }
    const int inputs[] = {hidden, ff1, hidden, hidden, hidden, hidden, hidden, hidden, hidden, ff2};
    const int outputs[] = {ff1, hidden, hidden, hidden, hidden, hidden, hidden * 2, hidden, ff2, hidden};
    for (int i = 0; i < 10; ++i) {
        const auto& linear = args->linears[i];
        if (!matrix(linear.weight, outputs[i], inputs[i], "conformer matrix") ||
            (linear.bias.data && !matrix(linear.bias, 1, outputs[i], "conformer bias"))) return 0;
        if (linear.has_clamp && (!std::isfinite(linear.in_min) || !std::isfinite(linear.in_max) ||
            !std::isfinite(linear.out_min) || !std::isfinite(linear.out_max) ||
            linear.in_min > linear.in_max || linear.out_min > linear.out_max)) {
            set_last_error("Invalid Gemma4 conformer projection clamp."); return 0;
        }
        if (linear.compressed_weight && (linear.compressed_type != GGML_TYPE_Q8_0 || inputs[i] % 32 != 0 ||
            linear.compressed_bytes != static_cast<int64_t>(ggml_row_size(GGML_TYPE_Q8_0, inputs[i])) * outputs[i])) {
            set_last_error("Gemma4 conformer compressed weights require exact Q8_0 rows and byte counts."); return 0;
        }
    }
    return true;
}

ggml_tensor * build_conformer_block(audio_graph& graph, ggml_tensor * x, const audio_block_args * args,
    std::vector<std::pair<ggml_tensor *, std::vector<float>>>& constants) {
    const int hidden = args->heads * args->head, sequence = args->sequence;
    const int ff1 = args->linears[0].weight.dim0, ff2 = args->linears[8].weight.dim0;
    const int inputs[] = {hidden, ff1, hidden, hidden, hidden, hidden, hidden, hidden, hidden, ff2};
    const int outputs[] = {ff1, hidden, hidden, hidden, hidden, hidden, hidden * 2, hidden, ff2, hidden};
    auto * ctx = graph.context.value;
    auto * relative = graph.bind(args->relative);
    ggml_tensor * weights[11] = {};
    ggml_tensor * matrices[10] = {};
    ggml_tensor * biases[10] = {};
    for (int i = 0; i < 11; ++i)
        if (args->weights[i].data) weights[i] = graph.bind_immutable(args->weights[i]);
    for (int i = 0; i < 10; ++i) {
        const auto& linear = args->linears[i];
        matrices[i] = linear.compressed_weight && !dense_projection_enabled(graph.require_float32_precision)
            ? graph.bind_quantized(linear.compressed_weight, GGML_TYPE_Q8_0, inputs[i], outputs[i], static_cast<size_t>(linear.compressed_bytes))
            : graph.bind_immutable(linear.weight);
        if (args->linears[i].bias.data) biases[i] = graph.bind_immutable(args->linears[i].bias);
    }
    auto clip = [&](ggml_tensor * input) { return ggml_clamp(ctx, input, -args->grad_clip, args->grad_clip); };
    auto norm = [&](ggml_tensor * input, int index) {
        // Missing affine norm means identity in the managed audio encoder.
        return weights[index] ? ggml_mul(ctx, ggml_rms_norm(ctx, input, args->epsilon), weights[index]) : input;
    };
    auto linear = [&](ggml_tensor * input, int index) {
        const auto& desc = args->linears[index];
        if (desc.has_clamp) input = ggml_clamp(ctx, input, desc.in_min, desc.in_max);
        auto * output = projection_mul(graph, matrices[index], input);
        if (biases[index]) output = ggml_add(ctx, output, biases[index]);
        return desc.has_clamp ? ggml_clamp(ctx, output, desc.out_min, desc.out_max) : output;
    };
    auto feed_forward = [&](ggml_tensor * input, int pre_norm, int post_norm, int up, int down) {
        auto * output = linear(ggml_silu(ctx, linear(norm(clip(input), pre_norm), up)), down);
        return ggml_add(ctx, input, ggml_scale(ctx, norm(clip(output), post_norm), args->residual_weight));
    };
    x = feed_forward(x, 0, 1, 0, 1);
    auto * residual = x;
    auto * normalized = norm(clip(x), 2);
    auto * q = ggml_scale(ctx, linear(normalized, 2), static_cast<float>(std::pow(args->head, -0.5) / std::log(2.0)));
    if (weights[9]) {
        q = ggml_reshape_2d(ctx, q, args->head, static_cast<int64_t>(args->heads) * sequence);
        q = ggml_mul(ctx, q, weights[9]);
        q = ggml_reshape_2d(ctx, q, hidden, sequence);
    }
    auto * k = ggml_scale(ctx, linear(normalized, 3), static_cast<float>(std::log(1.0 + std::exp(1.0)) / std::log(2.0)));
    auto * v = linear(normalized, 4);
    auto * attended = build_attention(graph, q, k, v, relative, sequence, args->heads, args->head,
        args->past, args->future, args->cap, constants);
    x = ggml_add(ctx, residual, norm(clip(linear(attended, 5)), 3));
    residual = x;
    // The convolution branch has no initial clipping, unlike FFN/attention.
    auto * pw1 = linear(norm(x, 4), 6);
    auto * data = ggml_cont(ctx, ggml_view_2d(ctx, pw1, hidden, sequence, pw1->nb[1], 0));
    auto * gate = ggml_cont(ctx, ggml_view_2d(ctx, pw1, hidden, sequence, pw1->nb[1], hidden * sizeof(float)));
    auto * convolution = ggml_mul(ctx, data, ggml_sigmoid(ctx, gate));
    if (weights[10]) convolution = build_depthwise(graph, convolution, weights[10], sequence, hidden, args->kernel, constants);
    x = ggml_add(ctx, residual, linear(ggml_silu(ctx, norm(clip(convolution), 5)), 7));
    x = feed_forward(x, 6, 7, 8, 9);
    x = norm(clip(x), 8);
    return x;
}

bool qualified_fused_stack(const audio_block_args * blocks, int count) {
    // Scope the measured default to the complete classic EmbeddingGemma 2
    // conformer. Other callers retain their existing explicit experiment.
    if (g_backend_type != BACKEND_TYPE_METAL || count != 12) return false;
    for (int i = 0; i < count; ++i) {
        const auto& block = blocks[i];
        if (block.require_float32_precision != 1 || block.heads != 8 || block.head != 128 ||
            block.past != 11 || block.future != 0 || block.kernel != 5 ||
            block.epsilon != 1e-6f || block.cap != 50.0f || block.grad_clip != 1e10f ||
            block.residual_weight != 0.5f || !block.weights[10].data ||
            block.linears[0].weight.dim0 != 4096 || block.linears[8].weight.dim0 != 4096)
            return false;
    }
    return true;
}

int conformer_stack(const audio_block_args * blocks, int count) {
    if (!blocks || count <= 0 || count > 128) {
        set_last_error("Gemma4 conformer stack requires 1 to 128 blocks."); return 0;
    }
    for (int i = 0; i < count; ++i) {
        if (!validate_block(blocks + i)) return 0;
        if (blocks[i].sequence != blocks[0].sequence || blocks[i].heads != blocks[0].heads || blocks[i].head != blocks[0].head) {
            set_last_error("Gemma4 conformer stack blocks must share their hidden shape."); return 0;
        }
    }
    bool require_float32_precision = false;
    for (int i = 0; i < count; ++i) require_float32_precision = require_float32_precision || blocks[i].require_float32_precision != 0;
    audio_graph graph(count, blocks[0].sequence, 10, require_float32_precision);
    graph.use_fused_operations = fused_ops_enabled(qualified_fused_stack(blocks, count));
    graph.bind(blocks[0].result);
    auto * x = graph.bind(blocks[0].input);
    std::vector<std::pair<ggml_tensor *, std::vector<float>>> constants;
    for (int i = 0; i < count; ++i) x = build_conformer_block(graph, x, blocks + i, constants);
    return graph.execute(x, constants);
}

int conformer_block(const audio_block_args * args) { return conformer_stack(args, 1); }

template<class Operation> int protect(Operation operation) {
    try { return operation(); }
    catch (const std::exception& error) { set_last_error(error.what()); return 0; }
    catch (...) { set_last_error("Unknown Gemma4 audio graph failure."); return 0; }
}
}

TSG_EXPORT int TSGgml_Gemma4AudioAttentionF32(TensorView2DDesc result, TensorView2DDesc query,
    TensorView2DDesc key, TensorView2DDesc value, TensorView2DDesc relative,
    int sequence, int heads, int head, int past, int future, float cap) {
    return protect([&] { return attention(result, query, key, value, relative, sequence, heads, head, past, future, cap); });
}

TSG_EXPORT int TSGgml_Gemma4AudioDepthwiseF32(TensorView2DDesc result, TensorView2DDesc input,
    TensorView2DDesc weight, int sequence, int channels, int kernel) {
    return protect([&] { return depthwise(result, input, weight, sequence, channels, kernel); });
}

TSG_EXPORT int TSGgml_Gemma4AudioConformerBlockF32(const audio_block_args * args) {
    return protect([&] { return conformer_block(args); });
}

TSG_EXPORT int TSGgml_Gemma4AudioConformerStackF32(const audio_block_args * blocks, int count) {
    return protect([&] { return conformer_stack(blocks, count); });
}

TSG_EXPORT int TSGgml_Gemma4AudioSscpF32(const audio_sscp_args * args) {
    return protect([&] { return sscp(args); });
}
