// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_deepseek41_tp.h"
#include "dsv41_workers.h"
#include "ggml_ops_dsv4_fused.h"
#include "ggml_ops_matmul_precision.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace tsg_dsv41_tp
{
namespace
{
void require(bool condition, const char * message)
{
    if (!condition) throw std::runtime_error(message);
}

// One read-only mapping is shared across the rank workers for a source
// tensor. Row-parallel quantized strips never require a full private copy.
class reader
{
public:
    explicit reader(const source & src) : descriptor(src)
    {
        require(src.ne[0] > 0 && src.ne[1] > 0 && src.ne[2] > 0 && src.ne[3] == 1,
                "Invalid V4.1 TP source dimensions");
        const size_t bytes = ggml_row_size(src.type, src.ne[0]) * src.ne[1] * src.ne[2];
#if !defined(_WIN32)
        int fd = open(src.path.c_str(), O_RDONLY);
        require(fd >= 0, "Cannot open V4.1 TP weight shard");
        struct stat status{};
        if (fstat(fd, &status) != 0 || status.st_size < 0 ||
            src.offset > (size_t) status.st_size || bytes > (size_t) status.st_size - src.offset)
        {
            close(fd);
            throw std::runtime_error("Truncated V4.1 TP weight source");
        }
        length = (size_t) status.st_size;
        mapping = mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        require(mapping != MAP_FAILED, "Cannot map V4.1 TP weight shard");
#else
        std::unique_ptr<FILE, decltype(&std::fclose)> pending(std::fopen(src.path.c_str(), "rb"), &std::fclose);
        require(pending != nullptr, "Cannot open V4.1 TP weight shard");
        require(_fseeki64(pending.get(), 0, SEEK_END) == 0, "Cannot seek V4.1 TP weight shard");
        const auto length = _ftelli64(pending.get());
        require(length >= 0 && src.offset <= (size_t) length && bytes <= (size_t) length - src.offset,
                "Truncated V4.1 TP weight source");
        file = pending.release();
#endif
    }
    ~reader()
    {
#if !defined(_WIN32)
        if (mapping != MAP_FAILED) munmap(mapping, length);
#else
        if (file) std::fclose(file);
#endif
    }
    void copy(void * destination, size_t offset, size_t bytes) const
    {
#if !defined(_WIN32)
        std::memcpy(destination, (const char *) mapping + descriptor.offset + offset, bytes);
#else
        std::lock_guard<std::mutex> lock(mutex);
        require(_fseeki64(file, descriptor.offset + offset, SEEK_SET) == 0 &&
                std::fread(destination, 1, bytes, file) == bytes, "Cannot read V4.1 TP weight strip");
#endif
    }
    const source & descriptor;
private:
#if !defined(_WIN32)
    void * mapping = MAP_FAILED;
    size_t length = 0;
#else
    FILE * file = nullptr;
    mutable std::mutex mutex;
#endif
};

void upload_strip(ggml_tensor * tensor, const reader & input, strip part, bool inner)
{
    const auto & src = input.descriptor;
    const size_t full_row = ggml_row_size(src.type, src.ne[0]);
    // Gate/up: contiguous strip of output rows per expert (GLM slice_mid).
    // Down: a quantization-block-aligned strip of every row (GLM slice_lo).
    const size_t run = inner ? ggml_row_size(src.type, part.count) : full_row * part.count;
    const size_t stride = inner ? full_row : full_row * src.ne[1];
    const size_t offset = inner ? ggml_row_size(src.type, part.first) : full_row * part.first;
    const int64_t count = inner ? src.ne[1] * src.ne[2] : src.ne[2];
    const size_t rows_per_batch = std::max<size_t>(1, (16 * 1024 * 1024) / run);
    std::vector<char> staging(std::min<size_t>(count, rows_per_batch) * run);
    for (int64_t start = 0; start < count; start += rows_per_batch)
    {
        const size_t rows = std::min<size_t>(count - start, rows_per_batch);
        for (size_t row = 0; row < rows; ++row)
            input.copy(staging.data() + row * run, offset + (start + row) * stride, run);
        ggml_backend_tensor_set(tensor, staging.data(), start * run, rows * run);
    }
}

struct graph
{
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * input = nullptr, * ids = nullptr, * weights = nullptr, * output = nullptr;
    ~graph() { if (ctx) ggml_free(ctx); }
};

struct rank_layer
{
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * gate = nullptr, * up = nullptr, * down = nullptr;
    std::map<int64_t, std::unique_ptr<graph>> graphs;
    ~rank_layer()
    {
        graphs.clear();
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
};

struct rank_state
{
    ggml_backend_t backend = nullptr;
    ggml_backend_t cuda_backend = nullptr;
    ggml_gallocr_t allocator = nullptr;
    std::map<int, std::unique_ptr<rank_layer>> layers;
    std::vector<float> partial;
    size_t weight_bytes = 0;
    ~rank_state()
    {
        layers.clear();
        if (allocator) ggml_gallocr_free(allocator);
        if (backend) ggml_backend_free(backend);
        if (cuda_backend) ggml_backend_free(cuda_backend);
    }
};
}

std::vector<strip> split(int64_t width, int64_t block, int ranks, int layer)
{
    // ggml CUDA's F32/BF16 vector kernels consume pairs, even though these
    // storage types have block size one. Quantized blocks are already even.
    block = block > 0 ? std::lcm<int64_t>(block, 2) : block;
    require(ranks >= 2 && ranks <= 16 && block > 0 && width > 0 &&
            width % block == 0 && width / block >= ranks, "Cannot form nonempty aligned V4.1 TP strips");
    const int64_t blocks = width / block, base = blocks / ranks, extra = blocks % ranks;
    std::vector<strip> result;
    int64_t first = 0;
    for (int rank = 0; rank < ranks; ++rank)
    {
        const int phase = ((rank - layer) % ranks + ranks) % ranks;
        const int64_t count = (base + (phase < extra)) * block;
        result.push_back({first, count});
        first += count;
    }
    return result;
}

std::vector<strip> split_weights(int64_t width, ggml_type down_type, int ranks, int layer)
{
    int64_t block = ggml_blck_size(down_type);
    // For BF16/F16, two-element alignment alone yields 330/328-wide strips
    // at 2304 channels on seven ranks. Those shapes change the upstream CUDA
    // matrix path and can amplify small projection differences when hidden
    // activations are rounded for the down projection. Aligned strips also
    // avoid the sorted-expert fallback's synchronization. No weights are padded
    // or converted, and small tensors keep the existing nonempty-strip policy.
    if ((down_type == GGML_TYPE_BF16 || down_type == GGML_TYPE_F16) &&
        width % 64 == 0 && width / 64 >= ranks)
        block = 64;
    return split(width, block, ranks, layer);
}

struct executor::impl
{
    struct layer
    {
        impl * owner = nullptr;
        int id = 0;
        int64_t embedding = 0;
        float clamp = 0;
    };
    int used;
    std::vector<std::unique_ptr<rank_state>> ranks;
    std::map<int, std::unique_ptr<layer>> layers;
    workers pool;
    std::string failure;
#if defined(TSG_GGML_TEST_HOOKS)
    int64_t test_position = 0;
    bool single_fanout = true;
    void test_fail(const char * stage, int layer, int rank)
    {
        const char * configured = std::getenv("TS_DSV41_TEST_FAIL_STAGE");
        const char * minimum = std::getenv("TS_DSV41_TEST_FAIL_POSITION");
        const char * selected_layer = std::getenv("TS_DSV41_TEST_FAIL_LAYER");
        const bool selected = configured && (std::strcmp(configured, stage) == 0 ||
            (std::strcmp(configured, "tp-rank-and-sync") == 0 &&
                (std::strcmp(stage, "tp-rank") == 0 || std::strcmp(stage, "tp-sync") == 0)));
        if (selected && rank == 1 &&
            layer == (selected_layer ? std::atoi(selected_layer) : 0) &&
            test_position >= (minimum ? std::atoll(minimum) : 0))
            throw std::runtime_error(std::string("Injected V4.1 TP failure at ") + stage);
    }
#endif

    impl(const std::vector<ggml_backend_dev_t> & devices, int used_experts)
        : used(used_experts), pool((int) devices.size())
    {
        require(devices.size() >= 2 && devices.size() <= 16 && used > 0,
                "V4.1 TP requires two to sixteen ranks and positive expert count");
        for (auto device : devices)
        {
            auto rank = std::make_unique<rank_state>();
            rank->backend = ggml_backend_dev_init(device, nullptr);
            require(rank->backend != nullptr, "Cannot initialize V4.1 TP rank backend");
#if defined(TSG_GGML_USE_CUDA)
            if (auto * wrapped = tsg_dsv4_fused_backend_init(rank->backend))
            {
                rank->cuda_backend = rank->backend;
                rank->backend = wrapped;
            }
#endif
            if (ggml_backend_is_cpu(rank->backend)) ggml_backend_cpu_set_n_threads(rank->backend, 1);
            rank->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(rank->backend));
            require(rank->allocator != nullptr, "Cannot initialize V4.1 TP scratch allocator");
            ranks.push_back(std::move(rank));
        }
    }

    graph & acquire(rank_state & rank, const layer & spec, int64_t tokens, int rank_id)
    {
        auto & weights = *rank.layers.at(spec.id);
        auto found = weights.graphs.find(tokens);
        std::unique_ptr<graph> pending;
        graph * slot = found == weights.graphs.end() ? nullptr : found->second.get();
        if (!slot)
        {
            pending = std::make_unique<graph>();
            slot = pending.get();
            auto & g = *slot;
            g.ctx = ggml_init({128 * ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false), nullptr, true});
            require(g.ctx != nullptr, "Cannot create V4.1 TP rank graph");
#if defined(TSG_GGML_TEST_HOOKS)
            test_fail("tp-graph", spec.id, rank_id);
#else
            (void) rank_id;
#endif
            g.gf = ggml_new_graph_custom(g.ctx, 128, false);
            g.input = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, spec.embedding, 1, tokens);
            g.ids = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, used, tokens);
            g.weights = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, 1, used, tokens);
            for (auto * input : {g.input, g.ids, g.weights}) ggml_set_input(input);
            auto matmul = [&](ggml_tensor * w, ggml_tensor * x) {
                if (w->type == GGML_TYPE_F32 && rank.cuda_backend)
                    return tsg_matmul_id_f32(g.ctx, w, x, g.ids);
                // Quantized strips take ggml's own integer (MMVQ/MMQ) paths,
                // which have no F32-precision variant: their activations are
                // requantized to Q8 per 32 values and the int8 products are
                // exact, so a strip already equals the same rows of the full
                // tensor up to F32 summation grouping. That grouping is
                // decided per launch (stream-k over the launch's tile count),
                // so bitwise agreement with an unsplit launch is not available
                // to any partition; dsv41_tp_test compares against the same
                // partition evaluated on one device where that matters.
                auto * out = ggml_mul_mat_id(g.ctx, w, x, g.ids);
                if (w->type == GGML_TYPE_F32)
                {
                    ggml_prec_set_acc(out, GGML_PREC_F32);
                    ggml_prec_set_src(out, GGML_PREC_F32, 1);
                }
                return out;
            };
            auto * up = matmul(weights.up, g.input);
            auto * gate = matmul(weights.gate, g.input);
            if (spec.clamp > 1e-6f)
            {
                up = ggml_clamp(g.ctx, up, -spec.clamp, spec.clamp);
                gate = ggml_clamp(g.ctx, gate, -INFINITY, spec.clamp);
            }
            auto * hidden = ggml_swiglu_split(g.ctx, gate, up);
            auto * experts = ggml_mul(g.ctx, matmul(weights.down, hidden), g.weights);
            for (int e = 0; e < used; ++e)
            {
                auto * value = ggml_view_2d(g.ctx, experts, spec.embedding, tokens, experts->nb[2], e * experts->nb[1]);
                g.output = g.output ? ggml_add(g.ctx, g.output, value) : value;
            }
            if (used == 1) g.output = ggml_cont(g.ctx, g.output);
            ggml_set_output(g.output);
            ggml_build_forward_expand(g.gf, g.output);
            for (int i = 0; i < ggml_graph_n_nodes(g.gf); ++i)
                require(ggml_backend_supports_op(rank.backend, ggml_graph_node(g.gf, i)),
                        "V4.1 TP rank cannot execute its local MoE graph");
        }
        // All layer graphs use one scratch allocation per rank. Clear only
        // this graph's temporary bindings; its weights live in separate contexts.
        for (auto * t = ggml_get_first_tensor(slot->ctx); t; t = ggml_get_next_tensor(slot->ctx, t))
        {
            t->data = nullptr;
            t->buffer = nullptr;
            t->extra = nullptr;
        }
        require(ggml_gallocr_alloc_graph(rank.allocator, slot->gf), "Cannot allocate V4.1 TP rank scratch");
        // Publish only a complete, allocated graph. A rank construction or
        // allocation exception must leave the same shape retryable later.
        if (pending)
        {
            if (weights.graphs.size() >= 4) weights.graphs.erase(weights.graphs.begin());
            weights.graphs.emplace(tokens, std::move(pending));
        }
        return *slot;
    }

    void compute(layer & spec, ggml_tensor * dst, const ggml_tensor * x,
                 const ggml_tensor * weights, const ggml_tensor * ids)
    {
        const int64_t tokens = x->ne[1], count = tokens * spec.embedding;
        auto submit = [&](int r) {
            auto & rank = *ranks[r];
            auto & g = acquire(rank, spec, tokens, r);
            rank.partial.resize(count);
            ggml_backend_tensor_set_async(rank.backend, g.input, x->data, 0, ggml_nbytes(g.input));
            ggml_backend_tensor_set_async(rank.backend, g.ids, ids->data, 0, ggml_nbytes(g.ids));
            ggml_backend_tensor_set_async(rank.backend, g.weights, weights->data, 0, ggml_nbytes(g.weights));
            require(ggml_backend_graph_compute_async(rank.backend, g.gf) == GGML_STATUS_SUCCESS,
                    "V4.1 TP rank graph execution failed");
            ggml_backend_tensor_get_async(rank.backend, g.output, rank.partial.data(), 0, count * sizeof(float));
#if defined(TSG_GGML_TEST_HOOKS)
            test_fail("tp-rank", spec.id, r);
#endif
        };
        auto synchronize = [&](int r) {
            ggml_backend_synchronize(ranks[r]->backend);
#if defined(TSG_GGML_TEST_HOOKS)
            // Throw only after draining the real queue, including when this
            // secondary error must not conceal a prior submission exception.
            test_fail("tp-sync", spec.id, r);
#endif
        };
#if defined(TSG_GGML_TEST_HOOKS)
        if (!single_fanout)
        {
            // Retain the previous dispatch path only for paired, same-binary
            // numerical and timing comparisons in the standalone test.
            try { pool.run(submit); }
            catch (...) { pool.run(synchronize); throw; }
            pool.run(synchronize);
        }
        else
#endif
        {
            // Every rank owns an independent queue. Let each worker submit
            // and drain its branch in one job; other workers keep submitting
            // concurrently. The pool joins all jobs before reduction/error
            // propagation, so no queued copy can outlive these host inputs.
            pool.run([&](int r) {
                std::exception_ptr first_error;
                try { submit(r); }
                catch (...) { first_error = std::current_exception(); }
                try { synchronize(r); }
                catch (...) { if (!first_error) first_error = std::current_exception(); }
                if (first_error) std::rethrow_exception(first_error);
            });
        }
        // Every branch ran concurrently. Host staging preserves F32 partials
        // and avoids transport-specific BF16 narrowing at the reduction boundary.
        auto * output = (float *) dst->data;
        std::copy(ranks[0]->partial.begin(), ranks[0]->partial.end(), output);
        for (size_t rank = 1; rank < ranks.size(); ++rank)
            for (int64_t i = 0; i < count; ++i) output[i] += ranks[rank]->partial[i];
    }

    static void callback(ggml_tensor * dst, const ggml_tensor * x, const ggml_tensor * weights,
                         const ggml_tensor * ids, int ith, int, void * opaque)
    {
        if (ith != 0) return;
        auto & spec = *(layer *) opaque;
        try { spec.owner->compute(spec, dst, x, weights, ids); }
        catch (const std::exception & error)
        {
            if (spec.owner->failure.empty()) spec.owner->failure = error.what();
            std::fill_n((float *) dst->data, ggml_nelements(dst), std::numeric_limits<float>::quiet_NaN());
        }
    }
};

executor::executor(const std::vector<ggml_backend_dev_t> & devices, int used_experts)
    : state(std::make_unique<impl>(devices, used_experts)) {}
executor::~executor() = default;

void executor::add_layer(int id, const source & gate, const source & up, const source & down, float clamp_limit)
{
    require(!has_layer(id), "V4.1 TP layer loaded twice");
    require(gate.ne == up.ne && gate.ne[0] == down.ne[1] && gate.ne[1] == down.ne[0] &&
            gate.ne[2] == down.ne[2] && gate.ne[3] == 1 && down.ne[3] == 1 &&
            state->used <= gate.ne[2], "V4.1 TP expert tensor shapes disagree");
    auto strips = split_weights(gate.ne[1], down.type, (int) state->ranks.size(), id);
    reader gate_file(gate), up_file(up), down_file(down);
    state->pool.run([&](int r) {
        auto & rank = *state->ranks[r];
        auto weights = std::make_unique<rank_layer>();
        weights->ctx = ggml_init({16 * ggml_tensor_overhead(), nullptr, true});
        require(weights->ctx != nullptr, "Cannot create V4.1 TP weight context");
        weights->gate = ggml_new_tensor_3d(weights->ctx, gate.type, gate.ne[0], strips[r].count, gate.ne[2]);
        weights->up = ggml_new_tensor_3d(weights->ctx, up.type, up.ne[0], strips[r].count, up.ne[2]);
        weights->down = ggml_new_tensor_3d(weights->ctx, down.type, strips[r].count, down.ne[1], down.ne[2]);
        weights->buffer = ggml_backend_alloc_ctx_tensors(weights->ctx, rank.backend);
        require(weights->buffer != nullptr, "Cannot allocate V4.1 TP weight strips");
        ggml_backend_buffer_set_usage(weights->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        upload_strip(weights->gate, gate_file, strips[r], false);
        upload_strip(weights->up, up_file, strips[r], false);
        upload_strip(weights->down, down_file, strips[r], true);
        rank.weight_bytes += ggml_nbytes(weights->gate) + ggml_nbytes(weights->up) + ggml_nbytes(weights->down);
        rank.layers.emplace(id, std::move(weights));
    });
    auto layer = std::make_unique<impl::layer>();
    layer->owner = state.get();
    layer->id = id;
    layer->embedding = gate.ne[0];
    layer->clamp = clamp_limit;
    state->layers.emplace(id, std::move(layer));
}

bool executor::has_layer(int layer) const { return state->layers.count(layer) != 0; }
size_t executor::rank_weight_bytes(int rank) const { return state->ranks.at(rank)->weight_bytes; }
void executor::begin_forward() { state->failure.clear(); }
std::string executor::error() const { return state->failure; }
#if defined(TSG_GGML_TEST_HOOKS)
void executor::test_set_position(int64_t position) { state->test_position = position; }
void executor::test_single_fanout(bool enabled) { state->single_fanout = enabled; }
#endif

ggml_tensor * executor::build(ggml_context * ctx, int layer, ggml_tensor * x, ggml_tensor * weights, ggml_tensor * ids)
{
    auto & spec = *state->layers.at(layer);
    require(x->type == GGML_TYPE_F32 && x->ne[0] == spec.embedding && x->ne[2] == 1 && x->ne[3] == 1 &&
            weights->type == GGML_TYPE_F32 && ggml_nelements(weights) == state->used * x->ne[1] &&
            ids->type == GGML_TYPE_I32 && ids->ne[0] == state->used && ids->ne[1] == x->ne[1],
            "Invalid V4.1 TP MoE inputs");
    // Unfused routing returns a top-k view with the original expert-count
    // column stride. Materialize it before the host broadcast reads bytes.
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    if (!ggml_is_contiguous(weights)) weights = ggml_cont(ctx, weights);
    if (!ggml_is_contiguous(ids)) ids = ggml_cont(ctx, ids);
    auto * output = ggml_map_custom3(ctx, x, weights, ids, impl::callback, 1, &spec);
    ggml_format_name(output, "v41_moe_tensor_parallel.%d", layer);
    return output;
}
}
