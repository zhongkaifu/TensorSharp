// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_graph_budget.h"
#include "ggml_ops_shared_cache_budget.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <thread>
#include <vector>

static void require(bool condition, const char* message)
{
    if (!condition) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

struct Ledger {
    struct Ticket { int rank; int64_t bytes; bool committed = false; };
    std::map<uint64_t, Ticket> tickets;
    int64_t capacity = 1 << 20, reserved = 0, committed = 0;
    std::atomic<int64_t> physical{0};
    uint64_t next = 0;
    bool fail_commit = false;
    static uint64_t reserve(void* context, int rank, int kind, int64_t bytes)
    {
        auto& self = *static_cast<Ledger*>(context);
        require(kind == 2 && rank >= 0 && rank <= 1, "incorrect graph kind/rank");
        if (bytes > self.capacity - self.reserved - self.committed) return 0;
        auto token = ++self.next;
        self.tickets.emplace(token, Ticket{rank, bytes});
        self.reserved += bytes;
        return token;
    }
    static int commit(void* context, uint64_t token)
    {
        auto& self = *static_cast<Ledger*>(context);
        if (self.fail_commit) return 0;
        auto& ticket = self.tickets.at(token);
        require(!ticket.committed, "duplicate commit");
        self.reserved -= ticket.bytes; self.committed += ticket.bytes;
        ticket.committed = true;
        return 1;
    }
    static void release(void* context, uint64_t token)
    {
        auto& self = *static_cast<Ledger*>(context);
        auto ticket = self.tickets.at(token);
        require(self.physical.load() <= self.reserved + self.committed - ticket.bytes,
            "refunded graph quota before physical free");
        (ticket.committed ? self.committed : self.reserved) -= ticket.bytes;
        self.tickets.erase(token);
    }
    bool attach(bool graphs = true)
    { return tsg::SharedCacheCharge::attach(this, reserve, commit, release, graphs); }
    bool detach() { return tsg::SharedCacheCharge::detach(this); }
    void empty()
    { require(tickets.empty() && reserved == 0 && committed == 0 && physical.load() == 0, "leaked graph owner"); }
};

// CPU-backed simulated allocator. Its native buffer/type/context/free identities
// are intentionally visible so a transparent wrapper must preserve all of them.
struct CpuType {
    struct Allocation { CpuType* owner; ggml_backend_buffer_t inner; };
    ggml_backend_buffer_type type{};
    Ledger& ledger;
    std::atomic<int> calls{0};
    int fail_at = -1;
    int actual_extra = 0;
    size_t max_size = 1 << 20;
    static CpuType& self(ggml_backend_buffer_type_t type) { return *static_cast<CpuType*>(type->context); }
    static const char* name(ggml_backend_buffer_type_t) { return "test-original-cpu-type"; }
    static size_t alignment(ggml_backend_buffer_type_t) { return 64; }
    static size_t maximum(ggml_backend_buffer_type_t type) { return self(type).max_size; }
    static bool host(ggml_backend_buffer_type_t) { return true; }
    static void* base(ggml_backend_buffer_t buffer)
    { return ggml_backend_buffer_get_base(static_cast<Allocation*>(buffer->context)->inner); }
    static void free_buffer(ggml_backend_buffer_t buffer)
    {
        auto* allocation = static_cast<Allocation*>(buffer->context);
        const size_t bytes = ggml_backend_buffer_get_size(allocation->inner);
        ggml_backend_buffer_free(allocation->inner);
        allocation->owner->ledger.physical.fetch_sub(static_cast<int64_t>(bytes));
        delete allocation;
    }
    static ggml_backend_buffer_t alloc(ggml_backend_buffer_type_t type, size_t bytes)
    {
        auto& owner = self(type);
        const int call = owner.calls.fetch_add(1) + 1;
        if (call == owner.fail_at) return nullptr;
        const size_t actual = static_cast<size_t>(static_cast<int64_t>(bytes) + owner.actual_extra);
        auto inner = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), actual);
        if (!inner) return nullptr;
        owner.ledger.physical.fetch_add(static_cast<int64_t>(actual));
        ggml_backend_buffer_i iface{};
        iface.free_buffer = free_buffer; iface.get_base = base;
        return ggml_backend_buffer_init(type, iface, new Allocation{&owner, inner}, actual);
    }
    explicit CpuType(Ledger& owner) : ledger(owner)
    {
        type.iface = {name, alloc, nullptr, alignment, maximum, nullptr, nullptr, host};
        type.context = this;
    }
};

struct Graph {
    ggml_context* ctx;
    ggml_cgraph* graph;
    ggml_tensor* input;
    ggml_tensor* output;
    explicit Graph(int elements)
    {
        ctx = ggml_init({1 << 20, nullptr, true});
        require(ctx != nullptr, "cannot allocate CPU test context");
        input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, elements);
        ggml_set_input(input);
        ggml_set_output(input); // keep the input alive; forbid in-place scale reuse
        output = ggml_scale(ctx, input, 2.0f);
        ggml_set_output(output);
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
    }
    ~Graph() { ggml_free(ctx); }
};

int main()
{
    Ledger ledger;
    CpuType type(ledger);
    auto buffer = tsg::graph_budget_alloc_buffer(&type.type, 64, 0);
    require(buffer && buffer->buft == &type.type && buffer->iface.free_buffer == CpuType::free_buffer,
        "wrapper changed backend identity");
    require(!ledger.attach(), "graph scope adopted an unbudgeted live graph");
    // The legacy cache-only API may coexist with graphs it explicitly excludes.
    require(ledger.attach(false) && ledger.detach(), "legacy cache-only scope changed behavior");
    tsg::graph_budget_free_buffer(buffer);
    ledger.empty();
    type.actual_extra = 1;
    buffer = tsg::graph_budget_alloc_buffer(&type.type, 64, 0);
    require(buffer && ggml_backend_buffer_get_size(buffer) == 65,
        "unconfigured graph adapter changed original allocator padding behavior");
    require(!ledger.attach(), "padded unconfigured graph escaped attach exclusion");
    tsg::graph_budget_free_buffer(buffer); ledger.empty(); type.actual_extra = 0;
    require(ledger.attach(), "graph budget attach failed");

    ledger.capacity = 63;
    const int calls = type.calls;
    require(!tsg::graph_budget_alloc_buffer(&type.type, 64, 0) && type.calls == calls,
        "physical allocation preceded reservation rejection");
    ledger.capacity = 1 << 20;
    type.fail_at = type.calls + 1;
    require(!tsg::graph_budget_alloc_buffer(&type.type, 64, 0), "allocator failure ignored");
    ledger.empty(); type.fail_at = -1;
    ledger.fail_commit = true;
    require(!tsg::graph_budget_alloc_buffer(&type.type, 64, 0), "commit failure ignored");
    ledger.empty(); ledger.fail_commit = false;
    type.actual_extra = 1;
    require(!tsg::graph_budget_alloc_buffer(&type.type, 64, 0), "under-reserved actual allocation published");
    ledger.empty(); type.actual_extra = -16;
    buffer = tsg::graph_budget_alloc_buffer(&type.type, 64, 0);
    require(buffer && ggml_backend_buffer_get_size(buffer) == 48 && ledger.committed == 64,
        "actual buffer smaller than bound corrupted accounting");
    require(!ledger.detach(), "detached callbacks with live graph owner");
    tsg::graph_budget_free_buffer(buffer); ledger.empty(); type.actual_extra = 0;

    // Real CPU context allocation also preserves native buffer type and can run.
    auto backend = ggml_backend_cpu_init();
    {
        Graph graph(32);
        buffer = tsg::graph_budget_alloc_ctx_tensors(graph.ctx, backend, 0);
        require(buffer && ledger.committed >= static_cast<int64_t>(ggml_backend_buffer_get_size(buffer)),
            "context allocation escaped graph ledger");
        require(buffer->buft == ggml_backend_get_default_buffer_type(backend), "context wrapper changed buft");
        auto* data = static_cast<float*>(graph.input->data);
        for (int i = 0; i < 32; ++i) data[i] = i * 0.25f;
        require(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS, "CPU graph failed");
        auto* result = static_cast<float*>(graph.output->data);
        for (int i = 0; i < 32; ++i) require(result[i] == i * 0.5f, "CPU context result changed");
        tsg::graph_budget_free_buffer(buffer);
    }
    ledger.empty(); ggml_backend_free(backend);

    type.max_size = 512;
    auto allocator = tsg::graph_budget_gallocr_new(&type.type, 1);
    require(allocator != nullptr, "gallocr construction failed");
    {
        Graph first(16);
        require(tsg::graph_budget_gallocr_alloc_graph(allocator, first.graph), "initial gallocr allocation failed");
        require(first.input->buffer->buft == &type.type
            && first.input->buffer->iface.free_buffer == CpuType::free_buffer, "gallocr changed buffer identity");
    }
    const int initial_calls = type.calls;
    const int64_t initial_charge = ledger.committed;
    {
        Graph same(16);
        require(tsg::graph_budget_gallocr_alloc_graph(allocator, same.graph), "gallocr reuse failed");
        require(type.calls == initial_calls && ledger.committed == initial_charge, "reuse allocated or double charged");
    }
    {
        Graph bigger(80);
        require(tsg::graph_budget_gallocr_alloc_graph(allocator, bigger.graph), "gallocr growth failed");
        require(ledger.reserved == 0 && ledger.committed == ledger.physical,
            "growth retained old generation charges or released new generation");
    }
    // Fail a later allocation in a new generation; partial chunks must be freed
    // before the corresponding owner tickets are returned. No pointer reuse test
    // can substitute for these allocator lifecycle boundaries.
    type.fail_at = type.calls + 2;
    {
        Graph failed(112);
        require(!tsg::graph_budget_gallocr_alloc_graph(allocator, failed.graph), "injected partial growth failure missing");
    }
    tsg::graph_budget_gallocr_free(allocator);
    ledger.empty(); type.fail_at = -1;

    ledger.capacity = 128;
    std::atomic<int> attempted{0}, admitted{0};
    std::atomic<bool> release{false};
    std::vector<std::thread> workers;
    for (int i = 0; i < 16; ++i) workers.emplace_back([&, i] {
        auto allocated = tsg::graph_budget_alloc_buffer(&type.type, 16, i % 2);
        if (allocated) ++admitted;
        ++attempted;
        while (!release.load()) std::this_thread::yield();
        tsg::graph_budget_free_buffer(allocated);
    });
    while (attempted.load() != 16) std::this_thread::yield();
    require(admitted == 8 && !ledger.detach(), "concurrent graph admission/detach escaped quota");
    release = true;
    for (auto& worker : workers) worker.join();
    ledger.empty();
    require(ledger.detach() && ledger.attach() && ledger.detach(), "released graph scope cannot detach/reattach");
    std::puts("PASS graph budget: original buffer ABI, reserve-before-alloc, actual-size rollback, physical-free release, gallocr generations, reuse and concurrent admission");
}
