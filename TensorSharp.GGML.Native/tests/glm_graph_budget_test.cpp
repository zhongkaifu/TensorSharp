// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Strict device-type identity and actual CPU scheduler allocation. CUDA model
// qualification is separate: this fixture cannot substitute for a device run.
#include "../glm_graph_budget.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <map>

namespace {
int checks = 0;
void require(bool value, const char* message) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL %s\n", message); std::exit(1); }
}
struct Ledger {
    std::int64_t used = 0, ceiling = 1 << 20;
    std::uint64_t next = 0;
    bool freed = false;
    std::map<std::uint64_t, std::int64_t> tickets;
    static std::uint64_t reserve(void* context, int rank, int kind, std::int64_t bytes) {
        auto& l = *static_cast<Ledger*>(context);
        require(rank == 0 && kind == 5, "exact graph rank and shared owner");
        if (bytes > l.ceiling - l.used) return 0;
        l.used += bytes; l.tickets[++l.next] = bytes; return l.next;
    }
    static int commit(void* context, std::uint64_t token) {
        return static_cast<Ledger*>(context)->tickets.count(token) ? 1 : 0;
    }
    static void release(void* context, std::uint64_t token) {
        auto& l = *static_cast<Ledger*>(context);
        require(l.freed, "physical scheduler free before graph credit release");
        l.used -= l.tickets.at(token); l.tickets.erase(token);
    }
};
struct StrictBackend {
    ggml_backend_t cpu;
    ggml_backend_buffer_type_t type;
    ggml_backend_device device{};
    ggml_backend backend{};
    int dispatched = 0, syncs = 0, frees = 0, copies = 0;
    static StrictBackend& dev(ggml_backend_dev_t d) { return *static_cast<StrictBackend*>(d->context); }
    static StrictBackend& stream(ggml_backend_t b) {
        auto& fixture = *static_cast<StrictBackend*>(b->context);
        require(b == &fixture.backend, "vendor dispatch receives its original backend pointer");
        return fixture;
    }
    static const char* name(ggml_backend_dev_t) { return "strict-CPU"; }
    static void memory(ggml_backend_dev_t, size_t* free, size_t* total) { *free = *total = 1 << 20; }
    static enum ggml_backend_dev_type device_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_CPU; }
    static void props(ggml_backend_dev_t d, struct ggml_backend_dev_props* p) {
        ggml_backend_dev_get_props(ggml_backend_get_device(dev(d).cpu), p);
    }
    static ggml_backend_t init(ggml_backend_dev_t, const char*) { return ggml_backend_cpu_init(); }
    static ggml_backend_buffer_type_t buffer_type(ggml_backend_dev_t d) { return dev(d).type; }
    static bool supports_op(ggml_backend_dev_t d, const ggml_tensor* t) { return ggml_backend_supports_op(dev(d).cpu, t); }
    static bool supports_type(ggml_backend_dev_t d, ggml_backend_buffer_type_t t) { return t == dev(d).type; }
    static const char* stream_name(ggml_backend_t b) { stream(b); return "strict-CPU"; }
    static void free_stream(ggml_backend_t b) { ++stream(b).frees; }
    static void sync(ggml_backend_t b) { auto& f = stream(b); ++f.syncs; ggml_backend_synchronize(f.cpu); }
    static enum ggml_status compute(ggml_backend_t b, ggml_cgraph* g) {
        auto& f = stream(b); ++f.dispatched; return ggml_backend_graph_compute_async(f.cpu, g);
    }
    static bool copy(ggml_backend_t src, ggml_backend_t dst, const ggml_tensor*, ggml_tensor*) {
        stream(src); ++stream(dst).copies; return true;
    }
    StrictBackend() : cpu(ggml_backend_cpu_init()), type(ggml_backend_cpu_buffer_type()) {
        require(cpu != nullptr, "actual CPU backend available");
        ggml_backend_cpu_set_n_threads(cpu, 1);
        device.iface = { name, name, memory, device_type, props, init, buffer_type,
            nullptr, nullptr, supports_op, supports_type, nullptr, nullptr, nullptr, nullptr };
        device.context = this;
        backend.guid = cpu->guid; backend.device = &device; backend.context = this;
        backend.iface.get_name = stream_name; backend.iface.free = free_stream;
        backend.iface.synchronize = sync; backend.iface.graph_compute = compute;
        backend.iface.cpy_tensor_async = copy;
    }
    ~StrictBackend() { ggml_backend_free(cpu); }
};
}
int main() {
    StrictBackend strict;
    Ledger ledger;
    require(tsg::SharedCacheCharge::attach(&ledger, Ledger::reserve, Ledger::commit, Ledger::release, true), "attach graph ledger");
    ggml_context* context = ggml_init({1 << 20, nullptr, true});
    require(context != nullptr, "graph metadata allocation");
    auto* input = ggml_new_tensor_1d(context, GGML_TYPE_F32, 32);
    ggml_set_input(input);
    auto* output = ggml_scale(context, input, 3.f);
    ggml_set_output(output);
    auto* graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    {
        tsg_glm::graph_budget budget;
        ggml_backend_buffer_type_t types[]{strict.type};
        ggml_backend_t originals[]{&strict.backend};
        auto* proxies = budget.prepare(types, originals, 1, 0, 5);
        require(proxies != nullptr && proxies[0] != types[0], "allocation descriptor has distinct identity");
        require(!ggml_backend_supports_buft(originals[0], proxies[0]), "strict vendor rejects bare allocation descriptor");
        auto alias = budget.scheduler_backend(originals[0]);
        require(alias && alias != originals[0] && alias == budget.scheduler_backends()[0], "exact scheduler facade identity");
        require(alias->guid == originals[0]->guid && alias->context == originals[0]->context, "vendor GUID and context preserved");
        require(ggml_backend_supports_buft(alias, proxies[0]), "facade unwraps its allocation descriptor");
        require(ggml_backend_supports_buft(alias, types[0]), "facade accepts original physical type");
        ggml_backend_buffer_type foreign = *types[0];
        require(!ggml_backend_supports_buft(alias, &foreign), "unknown lookalike type is rejected");
        require(budget.scheduler_backend(strict.cpu) == nullptr, "unknown backend cannot acquire a scheduler ID");
        require(ggml_backend_get_default_buffer_type(alias) == types[0], "default device type remains original");
        require(std::string(ggml_backend_name(alias)) == "strict-CPU", "name callback forwards original stream");
        StrictBackend second;
        tsg_glm::graph_backend second_facade(&second.backend, nullptr,
            [](void*, ggml_backend_buffer_type_t t) { return t; });
        require(alias->iface.cpy_tensor_async(second_facade.get(), alias, nullptr, nullptr), "copy unwraps both backend aliases");
        require(second.copies == 0 && strict.copies == 1, "destination backend owns the copy callback");
        require(second_facade.get()->iface.cpy_tensor_async(alias, &second.backend, nullptr, nullptr), "copy accepts mixed alias and original stream");
        require(second.copies == 1, "mixed copy reaches original destination stream");
        auto* scheduler = ggml_backend_sched_new(budget.scheduler_backends(), proxies, 1, 2048, false, false);
        require(scheduler != nullptr, "actual scheduler accepts strict facade/type pair");
        ggml_backend_sched_set_tensor_backend(scheduler, input, alias);
        ggml_backend_sched_set_tensor_backend(scheduler, output, alias);
        require(ggml_backend_sched_get_tensor_backend(scheduler, input) == alias, "input pin uses scheduler identity");
        require(ggml_backend_sched_get_tensor_backend(scheduler, output) == alias, "operation pin uses scheduler identity");
        require(ggml_backend_sched_alloc_graph(scheduler, graph), "budgeted actual CPU graph allocation");
        require(ledger.used > 0 && !tsg::SharedCacheCharge::detach(&ledger), "physical graph blocks ledger detach");
        require(ggml_backend_buffer_get_type(input->buffer) == types[0], "physical buffer retains original vendor type");
        float values[32], result[32];
        for (int i = 0; i < 32; ++i) values[i] = float(i) - 7.f;
        ggml_backend_tensor_set(input, values, 0, sizeof(values));
        require(ggml_backend_sched_graph_compute_async(scheduler, graph) == GGML_STATUS_SUCCESS, "actual scheduler compute through vendor dispatch");
        ggml_backend_sched_synchronize(scheduler);
        ggml_backend_tensor_get(output, result, 0, sizeof(result));
        for (int i = 0; i < 32; ++i) require(result[i] == 3.f * values[i], "actual CPU graph numerical result");
        require(strict.dispatched > 0 && strict.syncs > 0, "real dispatch and synchronization callbacks forwarded");
        ggml_backend_sched_free(scheduler);
        ledger.freed = true;
        require(ledger.used > 0, "scheduler physical free precedes owning descriptor refund");
        alias->iface.free(alias);
        require(strict.frees == 0, "graph alias never frees model-owned stream");
    }
    require(ledger.used == 0 && ledger.tickets.empty(), "every graph ticket returned");
    require(tsg::SharedCacheCharge::detach(&ledger), "empty graph ledger detaches");
    ggml_free(context);
    std::printf("PASS %d strict vendor-type and actual CPU scheduler checks; CUDA model acceptance remains separate\n", checks);
}
