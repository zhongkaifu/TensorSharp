// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_deepseek41_tp.h"
#include "dsv41_workers.h"
#include "dsv4_file_warm.h"
#include "ggml_ops_dsv4_fused.h"
#include "ggml_ops_matmul_precision.h"
#include "ggml_ops_tp_collective.h"
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
    const void * mapped_data() const
    {
#if !defined(_WIN32)
        return (const char *) mapping + descriptor.offset;
#else
        return nullptr;
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

void warm_layer(int layer, const reader & gate, const reader & up, const reader & down, int threads)
{
#if !defined(_WIN32)
    if (threads <= 0) return;
    // Network-backed mmap faults issue small synchronous requests, and the
    // rank strips repeatedly jump over other ranks' rows. Use the same bounded
    // sequential pread helper as ordinary DeepSeek loading, only for this
    // layer. All source mappings exist before the helper opens its descriptors,
    // so FUSE open-time cache invalidation cannot discard newly warmed pages.
    std::vector<std::string> paths;
    std::vector<tsg_dsv4::file_warm_range> ranges;
    for (const reader * input : {&gate, &up, &down})
    {
        const auto & src = input->descriptor;
        auto found = std::find(paths.begin(), paths.end(), src.path);
        const int file = (int) (found - paths.begin());
        if (found == paths.end()) paths.push_back(src.path);
        const size_t bytes = ggml_row_size(src.type, src.ne[0]) * src.ne[1] * src.ne[2];
        ranges.push_back({file, src.offset, bytes, input->mapped_data()});
    }
    tsg_dsv4::file_warm_options options;
    options.threads = threads;
    options.populate = true;
    const auto result = tsg_dsv4::warm_file_ranges(paths, ranges, options);
    require(result.ok, result.error.c_str());
    if (result.budget_skipped) return; // No alternate prefault; upload demand reads remain unchanged.
    if (result.bytes_total >= 1024ull * 1024 * 1024)
        std::fprintf(stderr, "[dsv41-tp] layer %d source warm: %.2f GiB in %.2fs (%d threads, %.2f GiB read, %.2f GiB resident)\n",
            layer, result.bytes_total / 1073741824.0, result.seconds, result.threads,
            result.bytes_read / 1073741824.0, result.bytes_resident / 1073741824.0);
#else
    (void) layer; (void) gate; (void) up; (void) down; (void) threads;
#endif
}

// Keep transfer storage alive across layers and forwards. CUDA's asynchronous
// tensor copies only overlap reliably when the host pages are pinned. Ask the
// unwrapped device for its host buffer type; ggml falls back to ordinary host
// allocation when pinning is unavailable, without changing numerical behavior.
struct host_staging
{
    ggml_backend_buffer_type_t type = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    size_t capacity = 0;
    ~host_staging() { if (buffer) ggml_backend_buffer_free(buffer); }
    char * reserve(size_t bytes)
    {
        if (capacity < bytes)
        {
            const size_t rounded = (bytes + 65535) & ~size_t(65535);
            auto * next = ggml_backend_buft_alloc_buffer(type ? type : ggml_backend_cpu_buffer_type(), rounded);
            require(next != nullptr, "Cannot allocate V4.1 TP host transfer buffer");
            if (buffer) ggml_backend_buffer_free(buffer);
            buffer = next;
            capacity = rounded;
        }
        return (char *) ggml_backend_buffer_get_base(buffer);
    }
};

void upload_strip(ggml_tensor * tensor, const reader & input, strip part, bool inner,
                  ggml_backend_t backend, host_staging & staging, bool pipelined = false)
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
    const size_t bank_bytes = std::min<size_t>(count, rows_per_batch) * run;
#if !defined(TSG_GGML_TEST_HOOKS)
    (void) backend; (void) staging; (void) pipelined;
#endif
#if defined(TSG_GGML_TEST_HOOKS)
    if (!pipelined)
#endif
    {
        std::vector<char> original(bank_bytes);
        for (int64_t start = 0; start < count; start += rows_per_batch)
        {
            const size_t rows = std::min<size_t>(count - start, rows_per_batch);
            for (size_t row = 0; row < rows; ++row)
                input.copy(original.data() + row * run, offset + (start + row) * stride, run);
            ggml_backend_tensor_set(tensor, original.data(), start * run, rows * run);
        }
        return;
    }
#if defined(TSG_GGML_TEST_HOOKS)
    // Retain the rejected upload experiment only for paired regression
    // measurements. The 40-layer trial was slower than pageable uploads.
    char * data = staging.reserve(2 * bank_bytes);
    int bank = 0;
    try
    {
        for (int64_t start = 0; start < count; start += rows_per_batch)
        {
            const size_t rows = std::min<size_t>(count - start, rows_per_batch);
            for (size_t row = 0; row < rows; ++row)
                input.copy(data + bank * bank_bytes + row * run, offset + (start + row) * stride, run);
            ggml_backend_tensor_set_async(backend, tensor, data + bank * bank_bytes, start * run, rows * run);
            // Filling the second bank overlaps the first bank's H2D transfer. Drain
            // both before either bank can be overwritten or the tensor can escape.
            if (bank == 1 || start + rows_per_batch >= count) ggml_backend_synchronize(backend);
            bank ^= 1;
        }
    }
    catch (...)
    {
        // A later file read can fail while the previous bank is in flight.
        // Its source buffer and destination tensor must outlive that transfer.
        ggml_backend_synchronize(backend);
        throw;
    }
#endif
}

struct graph
{
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr, * gate_graph = nullptr, * down_graph = nullptr;
    ggml_tensor * input = nullptr, * ids = nullptr, * weights = nullptr, * output = nullptr;
    ggml_tensor * hidden = nullptr, * hidden_input = nullptr;
    tsg_dsv4_fused_desc quant_strip, down_strip;
    ~graph() { if (ctx) ggml_free(ctx); }
};

struct rank_layer
{
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * gate = nullptr, * up = nullptr, * down = nullptr;
    int64_t full_rows = 0, first_row = 0, first_output = 0;
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
    host_staging transfer;
    float * result = nullptr;
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

std::vector<strip> split_outputs(int64_t width, int ranks, int layer)
{
    const int64_t alignment = width % 128 == 0 && width / 128 >= ranks ? 128 : 2;
    return split(width, alignment, ranks, layer);
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
    int load_threads;
    int host_gather_tokens = 0;
    std::vector<std::unique_ptr<rank_state>> ranks;
    std::map<int, std::unique_ptr<layer>> layers;
    workers pool;
    std::string failure;
    host_staging hidden_transfer;
    std::vector<float> pageable_hidden;
#if defined(TSG_GGML_USE_CUDA)
    tsg::TpF32Gather * device_gather = nullptr;
#endif
#if defined(TSG_GGML_TEST_HOOKS)
    int64_t test_position = 0;
    bool single_fanout = true;
    bool pinned_staging = true;
    bool pipelined_upload = false;
    int device_gather_override = -1;
    bool gate_fence = false;
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

    impl(const std::vector<ggml_backend_dev_t> & devices, int used_experts, int reader_threads)
        : used(used_experts), load_threads(reader_threads), pool((int) devices.size())
    {
        require(devices.size() >= 2 && devices.size() <= 16 && used > 0,
                "V4.1 TP requires two to sixteen ranks and positive expert count");
        const bool automatic_host_gather = std::getenv("TS_DSV41_TP_HOST_TOKENS") == nullptr;
        if (const char * configured = std::getenv("TS_DSV41_TP_HOST_TOKENS"))
        {
            char * end = nullptr;
            const long threshold = std::strtol(configured, &end, 10);
            require(end != configured && *end == '\0' && threshold >= 0 && threshold <= 4096,
                    "TS_DSV41_TP_HOST_TOKENS must be an integer from 0 to 4096");
            host_gather_tokens = int(threshold);
        }
        hidden_transfer.type = ggml_backend_dev_host_buffer_type(devices.front());
        for (auto device : devices)
        {
            auto rank = std::make_unique<rank_state>();
            rank->transfer.type = ggml_backend_dev_host_buffer_type(device);
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
#if defined(TSG_GGML_USE_CUDA)
        std::vector<ggml_backend_t> backends;
        for (const auto & rank : ranks) if (rank->cuda_backend) backends.push_back(rank->cuda_backend);
        if (backends.size() == ranks.size())
            device_gather = tsg::tp_cuda_f32_gather_create(backends.data(), (int) backends.size());
        if (automatic_host_gather)
        {
            // The factory can select the safe NCCL transport after probing.
            // Apply the measured small-batch policy to its final choice.
            const char * p2p_disabled = std::getenv("NCCL_P2P_DISABLE");
            host_gather_tokens = p2p_disabled && std::strcmp(p2p_disabled, "1") == 0 ? 16 : 0;
        }
        if (device_gather && host_gather_tokens)
            std::fprintf(stderr, "[dsv41-tp] pinned host activation gather for batches <= %d tokens; private F32 NCCL above.\n", host_gather_tokens);
#else
        (void) automatic_host_gather;
#endif
    }

    ~impl()
    {
#if defined(TSG_GGML_USE_CUDA)
        tsg::tp_cuda_f32_gather_free(device_gather);
#endif
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
            g.ctx = ggml_init({128 * ggml_tensor_overhead() + 3 * ggml_graph_overhead_custom(128, false), nullptr, true});
            require(g.ctx != nullptr, "Cannot create V4.1 TP rank graph");
#if defined(TSG_GGML_TEST_HOOKS)
            test_fail("tp-graph", spec.id, rank_id);
#else
            (void) rank_id;
#endif
            g.gf = ggml_new_graph_custom(g.ctx, 128, false);
            g.gate_graph = ggml_new_graph_custom(g.ctx, 128, false);
            g.down_graph = ggml_new_graph_custom(g.ctx, 128, false);
            g.input = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, spec.embedding, 1, tokens);
            g.ids = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, used, tokens);
            g.weights = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, 1, used, tokens);
            g.hidden_input = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, weights.full_rows, used, tokens);
            for (auto * input : {g.input, g.ids, g.weights, g.hidden_input}) ggml_set_input(input);
            g.quant_strip.kind = TSG_MATMUL_ID_QUANT_STRIP;
            g.quant_strip.i0 = int32_t(weights.full_rows);
            g.quant_strip.i1 = int32_t(weights.first_row);
            auto matmul = [&](ggml_tensor * w, ggml_tensor * x) {
#if defined(TSG_GGML_USE_CUDA)
                if (x == g.input && rank.cuda_backend && tsg_matmul_id_quant_strip_supported(
                        rank.cuda_backend, w, tokens, weights.full_rows, weights.first_row))
                    return tsg_matmul_id_quant_strip(g.ctx, w, x, g.ids, &g.quant_strip);
#endif
                if (w->type == GGML_TYPE_F32 && rank.cuda_backend)
                    return tsg_matmul_id_f32(g.ctx, w, x, g.ids);
                // Quantized strips that tsg_matmul_id_quant_strip_supported
                // accepts took the owned strip kernel above: it keeps the
                // unsplit launch's stream-k partitions and reduction order, so
                // its gate/up rows equal the full tensor's bit for bit. The
                // remaining quantized shapes use ggml's own integer (MMVQ/MMQ)
                // paths, which requantize activations to Q8 per 32 values and
                // decide F32 summation grouping per launch; dsv41_tp_test keeps
                // the full-weight reference as the pass criterion and records
                // the same-device partitioned evaluation beside it.
                auto * out = ggml_mul_mat_id(g.ctx, w, x, g.ids);
                if (w->type == GGML_TYPE_F32)
                {
                    ggml_prec_set_acc(out, GGML_PREC_F32);
                    ggml_prec_set_src(out, GGML_PREC_F32, 1);
                }
                return out;
            };
            ggml_tensor * up, * gate;
#if defined(TSG_GGML_USE_CUDA)
            if (rank.cuda_backend && weights.gate->type == weights.up->type &&
                tsg_matmul_id_quant_strip_supported(rank.cuda_backend, weights.gate, tokens, weights.full_rows, weights.first_row)) {
                g.quant_strip.kind = TSG_MATMUL_ID_QUANT_PAIR;
                auto * pair = tsg_matmul_id_quant_pair(g.ctx,weights.gate,weights.up,g.input,g.ids,&g.quant_strip);
                gate = ggml_view_3d(g.ctx,pair,pair->ne[0],pair->ne[1],pair->ne[2],pair->nb[1],pair->nb[2],0);
                up = ggml_view_3d(g.ctx,pair,pair->ne[0],pair->ne[1],pair->ne[2],pair->nb[1],pair->nb[2],pair->nb[3]);
            } else
#endif
            { up = matmul(weights.up,g.input); gate = matmul(weights.gate,g.input); }
            if (spec.clamp > 1e-6f)
            {
                up = ggml_clamp(g.ctx, up, -spec.clamp, spec.clamp);
                gate = ggml_clamp(g.ctx, gate, -INFINITY, spec.clamp);
            }
            g.hidden = ggml_swiglu_split(g.ctx, gate, up);
            ggml_set_output(g.hidden);
            ggml_build_forward_expand(g.gate_graph, g.hidden);
            ggml_build_forward_expand(g.gf, g.hidden);
            // Shard the down projection's output rows, retaining its complete
            // reduction dimension. Tiny split-K rounding changes can cross a
            // later activation quantizer's boundary and change model logits.
            ggml_tensor * down = nullptr;
#if defined(TSG_GGML_USE_CUDA)
            if (rank.cuda_backend && tsg_matmul_id_quant_strip_supported(rank.cuda_backend,
                    weights.down, tokens, spec.embedding, weights.first_output))
            {
                g.down_strip.kind = TSG_MATMUL_ID_QUANT_STRIP;
                g.down_strip.i0 = int32_t(spec.embedding);
                g.down_strip.i1 = int32_t(weights.first_output);
                down = tsg_matmul_id_quant_strip(g.ctx, weights.down, g.hidden_input, g.ids, &g.down_strip);
            }
#endif
            if (!down) down = matmul(weights.down, g.hidden_input);
            auto * experts = ggml_mul(g.ctx, down, g.weights);
            for (int e = 0; e < used; ++e)
            {
                auto * value = ggml_view_2d(g.ctx, experts, weights.down->ne[1], tokens, experts->nb[2], e * experts->nb[1]);
                g.output = g.output ? ggml_add(g.ctx, g.output, value) : value;
            }
            if (used == 1) g.output = ggml_cont(g.ctx, g.output);
            ggml_set_output(g.output);
            ggml_build_forward_expand(g.down_graph, g.output);
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
        const int64_t tokens = x->ne[1];
        const int64_t full_hidden = ranks.front()->layers.at(spec.id)->full_rows;
        const size_t hidden_count = size_t(full_hidden) * used * tokens;
        bool gather_on_device = false;
#if defined(TSG_GGML_USE_CUDA)
        gather_on_device = device_gather != nullptr && tokens > host_gather_tokens;
#if defined(TSG_GGML_TEST_HOOKS)
        if (device_gather_override >= 0)
            gather_on_device = device_gather != nullptr && device_gather_override != 0;
#endif
#endif
        float * gathered = nullptr;
        if (!gather_on_device)
        {
#if defined(TSG_GGML_TEST_HOOKS)
        if (!pinned_staging)
        {
            pageable_hidden.resize(hidden_count);
            gathered = pageable_hidden.data();
        }
        else
#endif
            gathered = (float *) hidden_transfer.reserve(hidden_count * sizeof(float));
        }
        auto synchronize = [&](int r) {
            ggml_backend_synchronize(ranks[r]->backend);
#if defined(TSG_GGML_TEST_HOOKS)
            test_fail("tp-sync", spec.id, r);
#endif
        };
        auto dispatch = [&](auto submit, bool wait = true) {
            if (!wait) { pool.run(submit); return; }
#if defined(TSG_GGML_TEST_HOOKS)
            if (!single_fanout)
            {
                try { pool.run(submit); }
                catch (...) { pool.run(synchronize); throw; }
                pool.run(synchronize);
                return;
            }
#endif
            pool.run([&](int r) {
                std::exception_ptr first_error;
                try { submit(r); }
                catch (...) { first_error = std::current_exception(); }
                try { synchronize(r); }
                catch (...) { if (!first_error) first_error = std::current_exception(); }
                if (first_error) std::rethrow_exception(first_error);
            });
        };
        // CUDA hidden exchange and the down graph use the same ordered raw
        // streams as gate/up. Only host exchange needs this intermediate fence;
        // the final output transfer is fenced before the CPU callback returns.
        bool wait_gate = !gather_on_device;
#if defined(TSG_GGML_TEST_HOOKS)
        wait_gate = wait_gate || gate_fence;
#endif
        dispatch([&](int r) {
            auto & rank = *ranks[r];
            auto & g = acquire(rank, spec, tokens, r);
            const void * input_data = x->data, * ids_data = ids->data, * weights_data = weights->data;
            const size_t result_bytes = std::max(ggml_nbytes(g.hidden), ggml_nbytes(g.output));
#if defined(TSG_GGML_TEST_HOOKS)
            if (!pinned_staging)
            {
                rank.partial.resize(result_bytes / sizeof(float));
                rank.result = rank.partial.data();
            }
            else
#endif
            {
                const size_t input_bytes = ggml_nbytes(g.input), ids_bytes = ggml_nbytes(g.ids),
                             weights_bytes = ggml_nbytes(g.weights);
                char * transfer = rank.transfer.reserve(input_bytes + ids_bytes + weights_bytes + result_bytes);
                input_data = transfer;
                ids_data = transfer + input_bytes;
                weights_data = transfer + input_bytes + ids_bytes;
                rank.result = (float *) (transfer + input_bytes + ids_bytes + weights_bytes);
                std::memcpy((void *) input_data, x->data, input_bytes);
                std::memcpy((void *) ids_data, ids->data, ids_bytes);
                std::memcpy((void *) weights_data, weights->data, weights_bytes);
            }
            ggml_backend_tensor_set_async(rank.backend, g.input, input_data, 0, ggml_nbytes(g.input));
            ggml_backend_tensor_set_async(rank.backend, g.ids, ids_data, 0, ggml_nbytes(g.ids));
            ggml_backend_tensor_set_async(rank.backend, g.weights, weights_data, 0, ggml_nbytes(g.weights));
            require(ggml_backend_graph_compute_async(rank.backend, g.gate_graph) == GGML_STATUS_SUCCESS,
                    "V4.1 TP gate/up graph execution failed");
            if (!gather_on_device)
                ggml_backend_tensor_get_async(rank.backend, g.hidden, rank.result, 0, ggml_nbytes(g.hidden));
#if defined(TSG_GGML_TEST_HOOKS)
            test_fail("tp-rank", spec.id, r);
#endif
        }, wait_gate);
        // Concatenate hidden rows independently for every selected expert/token.
        // No floating arithmetic or precision conversion crosses this boundary.
        if (gather_on_device)
        {
#if defined(TSG_GGML_USE_CUDA)
            std::vector<ggml_tensor *> sources, destinations;
            std::vector<int64_t> first, rows;
            for (auto & rank : ranks)
            {
                const auto & weight = *rank->layers.at(spec.id);
                const auto & graph = *weight.graphs.at(tokens);
                sources.push_back(graph.hidden); destinations.push_back(graph.hidden_input);
                first.push_back(weight.first_row); rows.push_back(weight.gate->ne[1]);
            }
            std::string error;
            const bool success = tsg::tp_cuda_f32_gather(device_gather, sources.data(), destinations.data(),
                first.data(), rows.data(), full_hidden, used * tokens, error);
            require(success, error.c_str());
#endif
        }
        else for (auto & rank : ranks)
        {
            const auto & weight = *rank->layers.at(spec.id);
            const int64_t rows = weight.gate->ne[1];
            for (int64_t column = 0; column < used * tokens; ++column)
                std::copy_n(rank->result + column * rows, rows,
                            gathered + column * full_hidden + weight.first_row);
        }
        dispatch([&](int r) {
            auto & rank = *ranks[r];
            auto & g = *rank.layers.at(spec.id)->graphs.at(tokens);
            if (!gather_on_device)
                ggml_backend_tensor_set_async(rank.backend, g.hidden_input, gathered, 0, ggml_nbytes(g.hidden_input));
            require(ggml_backend_graph_compute_async(rank.backend, g.down_graph) == GGML_STATUS_SUCCESS,
                    "V4.1 TP down graph execution failed");
            ggml_backend_tensor_get_async(rank.backend, g.output, rank.result, 0, ggml_nbytes(g.output));
#if defined(TSG_GGML_TEST_HOOKS)
            test_fail("tp-down-rank", spec.id, r);
#endif
        });
        // Output rows are disjoint. Gather them without reducing partial sums.
        auto * output = (float *) dst->data;
        for (auto & rank : ranks)
        {
            const auto & weight = *rank->layers.at(spec.id);
            const int64_t rows = weight.down->ne[1];
            for (int64_t token = 0; token < tokens; ++token)
                std::copy_n(rank->result + token * rows, rows,
                            output + token * spec.embedding + weight.first_output);
        }
    }

    static void callback(ggml_tensor * dst, const ggml_tensor * x, const ggml_tensor * weights,
                         const ggml_tensor * ids, int ith, int, void * opaque)
    {
        if (ith != 0) return;
        auto & spec = *(layer *) opaque;
        try { spec.owner->compute(spec, dst, x, weights, ids); }
        catch (const std::exception & error)
        {
            // The stream-ordered gate phase can leave work in
            // flight when a later graph/gather/submit operation fails. Drain
            // every rank without replacing the original request error.
            try { spec.owner->pool.run([&](int rank) {
                ggml_backend_synchronize(spec.owner->ranks[rank]->backend);
            }); } catch (...) {}
            if (spec.owner->failure.empty()) spec.owner->failure = error.what();
            std::fill_n((float *) dst->data, ggml_nelements(dst), std::numeric_limits<float>::quiet_NaN());
        }
    }
};

executor::executor(const std::vector<ggml_backend_dev_t> & devices, int used_experts, int load_threads)
    : state(std::make_unique<impl>(devices, used_experts, load_threads)) {}
executor::~executor() = default;

void executor::add_layer(int id, const source & gate, const source & up, const source & down, float clamp_limit)
{
    require(!has_layer(id), "V4.1 TP layer loaded twice");
    require(gate.ne == up.ne && gate.ne[0] == down.ne[1] && gate.ne[1] == down.ne[0] &&
            gate.ne[2] == down.ne[2] && gate.ne[3] == 1 && down.ne[3] == 1 &&
            state->used <= gate.ne[2], "V4.1 TP expert tensor shapes disagree");
    auto strips = split_weights(gate.ne[1], down.type, (int) state->ranks.size(), id);
    auto output_strips = split_outputs(down.ne[1], (int) state->ranks.size(), id);
    reader gate_file(gate), up_file(up), down_file(down);
    warm_layer(id, gate_file, up_file, down_file, state->load_threads);
    state->pool.run([&](int r) {
        auto & rank = *state->ranks[r];
        auto weights = std::make_unique<rank_layer>();
        weights->full_rows = gate.ne[1];
        weights->first_row = strips[r].first;
        weights->first_output = output_strips[r].first;
        weights->ctx = ggml_init({16 * ggml_tensor_overhead(), nullptr, true});
        require(weights->ctx != nullptr, "Cannot create V4.1 TP weight context");
        weights->gate = ggml_new_tensor_3d(weights->ctx, gate.type, gate.ne[0], strips[r].count, gate.ne[2]);
        weights->up = ggml_new_tensor_3d(weights->ctx, up.type, up.ne[0], strips[r].count, up.ne[2]);
        weights->down = ggml_new_tensor_3d(weights->ctx, down.type, down.ne[0], output_strips[r].count, down.ne[2]);
        weights->buffer = ggml_backend_alloc_ctx_tensors(weights->ctx, rank.backend);
        require(weights->buffer != nullptr, "Cannot allocate V4.1 TP weight strips");
        ggml_backend_buffer_set_usage(weights->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        bool pipeline = false;
#if defined(TSG_GGML_TEST_HOOKS)
        pipeline = state->pipelined_upload;
#endif
        upload_strip(weights->gate, gate_file, strips[r], false, rank.backend, rank.transfer, pipeline);
        upload_strip(weights->up, up_file, strips[r], false, rank.backend, rank.transfer, pipeline);
        upload_strip(weights->down, down_file, output_strips[r], false, rank.backend, rank.transfer, pipeline);
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
void executor::test_pinned_staging(bool enabled) { state->pinned_staging = enabled; }
void executor::test_pipelined_upload(bool enabled) { state->pipelined_upload = enabled; }
void executor::test_device_gather(bool enabled)
{
#if defined(TSG_GGML_USE_CUDA)
    require(!enabled || state->device_gather != nullptr, "Device gather benchmark requires an available private CUDA collective");
#else
    require(!enabled, "Device gather benchmark requires CUDA");
#endif
    state->device_gather_override = enabled ? 1 : 0;
}
void executor::test_gate_fence(bool enabled) { state->gate_fence = enabled; }
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
