// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../dsv4_host_expert_backend.h"
#include "ggml-cpu.h"
#include <cassert>
#include <iostream>

static void require(bool ok, const char * message) {
    if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}
#if defined(__linux__)
static void registered_matmuls() {
    using namespace tsg_dsv4;
    char path[] = "/tmp/ts-registered-expert-XXXXXX";
    const int fd = mkstemp(path);
    require(fd >= 0, "Cannot create registered-expert fixture");
    std::vector<uint8_t> bytes(12288, 0);
    for (int p = 0; p < 3; ++p) for (int e = 0; e < 5; ++e) for (int i = 0; i < 12; ++i) {
        const float value = float(p * 20 + e + 1);
        memcpy(bytes.data() + p * 4096 + (e * 12 + i) * sizeof(float), &value, sizeof(value));
    }
    require(write(fd, bytes.data(), bytes.size()) == ssize_t(bytes.size()), "Cannot write expert fixture");
    auto * mapped = static_cast<uint8_t *>(mmap(nullptr, bytes.size(), PROT_READ, MAP_SHARED, fd, 0));
    require(mapped != MAP_FAILED, "Cannot map registered-expert fixture");
    {
        host_expert_reader reader({path}, 2, 8192);
        for (int prefix : {0, 1}) for (int threads : {1, 4, 16}) for (int columns : {1, 3}) {
            auto * ctx = ggml_init({1024 * 1024, nullptr, false});
            auto * cpu = ggml_backend_cpu_init();
            ggml_backend_cpu_set_n_threads(cpu, threads);
            auto * backend = host_expert_backend(cpu);
            host_expert_read_context context;
            context.reader = &reader; context.experts = 5;
            ggml_tensor * weights[3];
            for (int p = 0; p < 3; ++p) {
                weights[p] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 4, 5);
                weights[p]->data = mapped + p * 4096;
                context.projections[p] = {0, uint64_t(p * 4096), 240, weights[p]->data};
            }
            host_expert_backend_register(backend, &context);
            auto * input = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 12);
            const int32_t ids[] = {4, 1, 0, 0, 1, 4, 0, 0, 4, 1, 0, 0};
            memcpy(input->data, ids, sizeof(ids));
            auto * selected = ggml_view_2d(ctx, input, 2, columns, 16, 0);
            if (prefix) selected = ggml_dup(ctx, selected);
            auto * activation = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 1, columns);
            for (int i = 0; i < 3 * columns; ++i) static_cast<float *>(activation->data)[i] = float(i % 3 + 1);
            auto * up = ggml_mul_mat_id(ctx, weights[0], activation, selected);
            auto * gate = ggml_mul_mat_id(ctx, weights[1], activation, selected);
            auto * output = ggml_add(ctx, up, gate);
            int foreign_calls = 0;
            ggml_tensor * args[] = {output};
            auto * foreign = ggml_custom_4d(ctx, GGML_TYPE_F32, 4, 2, columns, 1, args, 1,
                [](ggml_tensor * dst, int ith, int, void * data) {
                    if (ith) return;
                    ++*static_cast<int *>(data);
                    memcpy(dst->data, dst->src[0]->data, ggml_nbytes(dst));
                }, 1, &foreign_calls);
            auto * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, foreign);
            const int nodes = graph->n_nodes;
            require(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS, "Reference expert graph failed");
            std::vector<float> reference(static_cast<float *>(output->data), static_cast<float *>(output->data) + 8 * columns);
            const bool adaptive_case = prefix == 1 && threads == 16 && columns == 1;
            for (int pass = 0; pass < (adaptive_case ? 41 : 9); ++pass) {
                // A previous reference run must not mask a missing CPU-prefix
                // dependency: the reader must see newly produced IDs.
                if (prefix) std::fill_n(static_cast<int32_t *>(selected->data), 2 * columns, -1);
                const auto before = reader.requested;
                require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Registered expert graph failed");
                require(graph->n_nodes == nodes && !reader.error()[0], "Wrapper changed graph topology or failed I/O");
                require(memcmp(output->data, reference.data(), reference.size() * sizeof(float)) == 0, "Expert logits differ from untouched CPU graph");
                for (int j = 0; j < columns; ++j) for (int e = 0; e < 2; ++e) for (int row = 0; row < 4; ++row)
                    require(static_cast<float *>(output->data)[j * 8 + e * 4 + row] == 12.0f * ids[j * 4 + e] + 132.0f,
                        "Expert graph disagrees with exact integer reference");
                const bool checked = (!adaptive_case || pass < 32) && (columns > 1 || pass % 8 == 0);
                require(reader.requested - before == (checked ? 288u : 0u), "Gate/up must share preparation and confirmed warm decode must bypass probes");
                require(foreign_calls == pass + 2 && memcmp(foreign->data, output->data, ggml_nbytes(output)) == 0,
                    "Foreign custom operation was skipped or changed");
            }
            if (adaptive_case) {
                auto & state = *static_cast<host_expert_backend_state *>(backend->context);
                require(state.feedback.hot() && state.bypassed_graphs == 9, "Real graph did not enter warm bypass");
                auto * wide_ids = ggml_view_2d(ctx, input, 2, 3, 16, 0);
                auto * wide_activation = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 1, 3);
                for (int i = 0; i < 9; ++i) static_cast<float *>(wide_activation->data)[i] = float(i % 3 + 1);
                auto * wide = ggml_mul_mat_id(ctx, weights[0], wide_activation, wide_ids);
                auto * prefill = ggml_new_graph(ctx);
                ggml_build_forward_expand(prefill, wide);
                const auto before = reader.requested;
                require(ggml_backend_graph_compute(backend, prefill) == GGML_STATUS_SUCCESS &&
                    !state.feedback.hot() && reader.requested - before == 288,
                    "Prefill must leave warm bypass and prepare current routes");
                for (int j = 0; j < 3; ++j) for (int e = 0; e < 2; ++e) for (int row = 0; row < 4; ++row)
                    require(static_cast<float *>(wide->data)[j * 8 + e * 4 + row] == 6.0f * (ids[j * 4 + e] + 1),
                        "Prefill after warm bypass changed arithmetic");
            }
            if (prefix == 1 && threads == 4 && columns == 1) {
                auto * other_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
                static_cast<int32_t *>(other_ids->data)[0] = 0;
                static_cast<int32_t *>(other_ids->data)[1] = 3;
                auto * other = ggml_mul_mat_id(ctx, weights[0], activation, other_ids);
                auto * combined = ggml_add(ctx, foreign, other);
                auto * distinct = ggml_new_graph(ctx);
                ggml_build_forward_expand(distinct, combined);
                require(ggml_backend_graph_compute(cpu, distinct) == GGML_STATUS_SUCCESS, "Distinct-route reference failed");
                std::vector<float> expected(static_cast<float *>(combined->data), static_cast<float *>(combined->data) + 8);
                context.hints = expert_residency_hints{};
                const auto before = reader.requested;
                require(ggml_backend_graph_compute(backend, distinct) == GGML_STATUS_SUCCESS &&
                    reader.requested - before == 576 &&
                    memcmp(combined->data, expected.data(), expected.size() * sizeof(float)) == 0,
                    "Different routed ID tensors must prepare separately and preserve arithmetic");
            }
            if (prefix == 1 && threads == 16 && columns == 3) {
                reader.fail("injected reader failure");
                require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED && foreign_calls == 10,
                    "A failed read must stop before later computation");
            }
            ggml_backend_free(backend); ggml_backend_free(cpu); ggml_free(ctx);
        }
    }
    require(memcmp(mapped, bytes.data(), bytes.size()) == 0, "Registered reads changed model bytes");
    munmap(mapped, bytes.size()); close(fd); unlink(path);
}
#endif
int main() {
    using namespace tsg_dsv4;
    expert_read_feedback feedback;
    for (int i = 0; i < 31; ++i) feedback.prepared(0);
    require(!feedback.ready(), "Unconfirmed warm streak must not bypass reads");
    feedback.prepared(0); feedback.arm(7);
    require(feedback.check(7), "Confirmed warm streak should bypass probes");
    require(!feedback.check(8) && !feedback.ready(), "Major faults must restore preparation");
    for (int i = 0; i < 32; ++i) feedback.prepared(0);
    feedback.arm(9); feedback.prepared(4096);
    require(!feedback.hot() && !feedback.ready(), "Actual I/O must cancel warm bypass");
    require(!host_expert_read_enabled(nullptr) && !host_expert_read_enabled("") &&
        !host_expert_read_enabled("0") && host_expert_read_enabled("1"), "Invalid read policy defaults");
    for (const char * value : {"2", "true", "-1", "1x"}) {
        bool refused = false;
        try { host_expert_read_enabled(value); } catch (const std::runtime_error &) { refused = true; }
        require(refused, "Malformed read policy must be refused");
    }
    expert_residency_hints hints;
    auto initial = hints.select({1, 1, 4}, 5, true);
    require(initial == std::vector<int32_t>({1, 1, 4}), "New experts must be checked");
    hints.mark(initial);
    for (int i = 0; i < 7; ++i) require(hints.select({1, 4}, 5, true).empty(), "Hot decode hint expired early");
    require(hints.select({1, 4}, 5, true).size() == 2, "Hot decode hint must expire");
    hints.mark({1, 4});
    require(hints.select({1, 2}, 5, true) == std::vector<int32_t>({2}), "New expert was skipped");
    require(hints.select({1, 4}, 5, false).size() == 2, "Prefill must always check residency");
    require(hints.select({2}, 5, true).size() == 1, "Unconfirmed read was marked hot");
    require(hints.select({1}, 3, true).size() == 1, "Changed expert dimension must reset hints");
    hints.mark({1}); hints.invalidate();
    require(hints.select({1}, 3, true).size() == 1, "Fault feedback must invalidate recent residency hints");
    std::vector<uint8_t> data(3 * 5 * 8192 + 3);
    for (size_t i = 0; i < data.size(); ++i) data[i] = uint8_t(i * 31 + i / 7);
    std::array<file_warm_range, 3> projections;
    for (size_t i = 0; i < 3; ++i) projections[i] = {0, i * 5 * 8192 + 1, 5 * 8192, data.data() + i * 5 * 8192 + 1};
    auto ranges = selected_expert_ranges(projections, 5, {4, 1, 4, 1});
    require(ranges.size() == 6, "Deduplicate exactly the selected projections");
    require(ranges[0].offset == 8193 && ranges[3].offset == 32769, "Expert offsets are wrong");
    require(selected_expert_ranges(projections, 5, {}).empty(), "Empty routing must read nothing");
    for (int bad : {-1, 5}) {
        bool refused = false;
        try { selected_expert_ranges(projections, 5, {1, bad}); }
        catch (const std::runtime_error &) { refused = true; }
        require(refused, "Invalid expert must be refused before I/O");
    }
#if defined(__linux__)
    char path[] = "/tmp/ts-demand-expert-XXXXXX";
    int fd = mkstemp(path);
    require(fd >= 0, "Cannot create fixture");
    require(write(fd, data.data(), data.size()) == ssize_t(data.size()) && fsync(fd) == 0, "Cannot write fixture");
    auto * mapped = static_cast<uint8_t *>(mmap(nullptr, data.size(), PROT_READ, MAP_SHARED, fd, 0));
    require(mapped != MAP_FAILED, "Cannot map fixture");
    for (size_t i = 0; i < 3; ++i) projections[i].mapped = mapped + projections[i].offset;
    ranges = selected_expert_ranges(projections, 5, {4, 1, 4});
    struct ledger { int64_t used = 0, limit = 16384; bool commit = true; } budget;
    const auto attach = [&] {
        return tsg::SharedCacheCharge::attach(&budget,
            [](void * c, int rank, int kind, int64_t bytes) -> uint64_t {
                auto & b = *static_cast<ledger *>(c);
                require(rank == 0 && kind == 3, "Staging must charge shared host pools");
                if (bytes > b.limit - b.used) return 0;
                b.used += bytes; return uint64_t(bytes);
            }, [](void * c, uint64_t) { return int(static_cast<ledger *>(c)->commit); },
            [](void * c, uint64_t token) { static_cast<ledger *>(c)->used -= int64_t(token); }, false, true);
    };
    require(attach(), "Cannot install staging budget fixture");
    {
        host_expert_reader reader({path}, 4, 4 * 4096);
        require(reader.staging_bytes() == 16384 && reader.threads() == 4, "Staging exceeds budget");
        require(budget.used == 16384 && !tsg::SharedCacheCharge::detach(&budget), "Live staging lost its shared charge");
        bool denied = false;
        try { host_expert_reader competing({path}, 1, 4096); }
        catch (const std::runtime_error &) { denied = true; }
        require(denied && budget.used == 16384, "Concurrent owner exceeded shared host quota");
        madvise(mapped, data.size(), MADV_DONTNEED);
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        reader.warm(ranges);
        require(!reader.error()[0], reader.error());
        require(memcmp(mapped, data.data(), data.size()) == 0, "Read changed mapped weight bytes");
        const auto read_bytes = reader.read;
        reader.warm(ranges);
        require(!reader.error()[0] && reader.read == read_bytes, "Resident ranges must not be re-read");
        require(reader.resident >= 6 * 8192, "Resident skip was not observed");
        auto bad = ranges; bad[0].offset = data.size();
        reader.warm(bad);
        require(reader.error()[0] && reader.read == read_bytes, "Out-of-file range must fail before any read");

    }
    require(budget.used == 0, "Destroyed staging retained its charge");
    budget.commit = false;
    bool denied = false;
    try { host_expert_reader failed_commit({path}, 1, 4096); }
    catch (const std::runtime_error &) { denied = true; }
    require(denied && budget.used == 0, "Failed commit leaked a reservation");
    require(tsg::SharedCacheCharge::detach(&budget), "Released staging still blocks detach");
    {
        host_expert_reader reader({path}, 2, 8192);
        require(!attach(), "Cannot attach around existing uncharged staging");
        require(ftruncate(fd, 0) == 0, "Cannot truncate test fixture");
        reader.warm(ranges);
        require(reader.error()[0], "Unexpected EOF must not produce a successful demand read");
    }
    munmap(mapped, data.size()); close(fd); unlink(path);
    registered_matmuls();
#endif
    std::cout << "host expert ranges, shared staging quota, resident skip, byte identity and I/O failure: pass\n";
}
