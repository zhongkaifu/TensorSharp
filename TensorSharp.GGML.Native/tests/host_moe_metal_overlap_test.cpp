// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Graph metadata and synthetic allocation ranges only; no backend or model.
#include "ggml_ops_host_moe_metal_overlap.h"
#include "ggml-impl.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
namespace ov = tsg::host_moe_metal_overlap;
int checks = 0;
void require(bool value, const char* message)
{
    ++checks;
    if (!value) throw std::runtime_error(message);
}
void* address(std::uintptr_t value) { return reinterpret_cast<void*>(value); }

struct Fixture {
    ggml_context* ctx = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor *x, *ids, *weights, *router, *pending, *gate, *up, *activation, *down, *shared, *join;
    std::array<const ggml_tensor*, 3> inputs;
    Fixture(int hidden = 4, int shared_ff = 4, int rows = 1, int used = 4, int experts = 8)
    {
        if (hidden <= 0 || shared_ff <= 0 || rows <= 0 || used <= 0 || used > experts)
            throw std::invalid_argument("invalid synthetic MoE geometry");
        ctx = ggml_init({2u << 20, nullptr, true});
        if (!ctx) throw std::runtime_error("metadata context allocation");
        auto matrix = [&](int input, int output) { return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input, output); };
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, rows);
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, rows);
        weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, used, rows);
        router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, experts, rows);
        pending = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, rows);
        for (auto* input : {x, ids, weights}) ggml_set_output(input);
        ggml_set_input(pending); ggml_set_output(pending);
        inputs = {x, ids, weights};
        gate = ggml_mul_mat(ctx, matrix(hidden, shared_ff), x);
        up = ggml_mul_mat(ctx, matrix(hidden, shared_ff), x);
        activation = ggml_mul(ctx, ggml_silu(ctx, gate), up);
        down = ggml_mul_mat(ctx, matrix(shared_ff, hidden), activation);
        auto* scalar = ggml_sigmoid(ctx, ggml_mul_mat(ctx, matrix(hidden, 1), x));
        shared = ggml_mul(ctx, down, scalar);
        join = ggml_add(ctx, pending, shared);
        graph = ggml_new_graph_custom(ctx, 128, false);
        ggml_build_forward_expand(graph, join);
        std::uintptr_t next = 0x10000000;
        for (auto* t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            if (!t->view_src) {
                t->data = address(next);
                next += ((ggml_nbytes(t) + 4095) / 4096 + 1) * 4096;
            }
        }
        for (auto* t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t))
            if (t->view_src) t->data = address(reinterpret_cast<std::uintptr_t>(t->view_src->data) + t->view_offs);
    }
    ~Fixture() { ggml_free(ctx); }
    Fixture(const Fixture&) = delete;
    int index(const ggml_tensor* tensor) const
    {
        for (int i = 0; i < graph->n_nodes; ++i) if (graph->nodes[i] == tensor) return i;
        return -1;
    }
    int cut(ov::Reject& why) { return ov::prefix_end(graph, 0, graph->n_nodes, pending, why); }
    bool safe(ov::Reject& why)
    {
        return ov::safe_allocations(graph, 0, index(join), pending, inputs.data(), static_cast<int>(inputs.size()), why);
    }
};

struct Snapshot {
    struct Tensor {
        ggml_tensor* pointer;
        std::array<int64_t, GGML_MAX_DIMS> ne;
        std::array<size_t, GGML_MAX_DIMS> nb;
        void* data;
        int flags;
    };
    std::vector<Tensor> tensors;
    std::vector<ggml_tensor*> nodes;
    explicit Snapshot(Fixture& fixture)
    {
        nodes.assign(fixture.graph->nodes, fixture.graph->nodes + fixture.graph->n_nodes);
        for (auto* t = ggml_get_first_tensor(fixture.ctx); t; t = ggml_get_next_tensor(fixture.ctx, t)) {
            Tensor copy{t, {}, {}, t->data, t->flags};
            std::copy(t->ne, t->ne + GGML_MAX_DIMS, copy.ne.begin());
            std::copy(t->nb, t->nb + GGML_MAX_DIMS, copy.nb.begin());
            tensors.push_back(copy);
        }
    }
    void unchanged(const Fixture& fixture) const
    {
        require(nodes.size() == static_cast<size_t>(fixture.graph->n_nodes) &&
            std::equal(nodes.begin(), nodes.end(), fixture.graph->nodes), "planner preserves full graph node order");
        for (const auto& t : tensors)
            require(t.data == t.pointer->data && t.flags == t.pointer->flags &&
                std::equal(t.ne.begin(), t.ne.end(), t.pointer->ne) &&
                std::equal(t.nb.begin(), t.nb.end(), t.pointer->nb),
                "planner preserves full graph shapes, strides, pins and allocations");
    }
};

void eligible_prefix()
{
    Fixture f; Snapshot before(f); ov::Reject why = ov::Reject::none;
    const int end = f.cut(why);
    require(end == f.index(f.join) && why == ov::Reject::none, "shared FFN stops immediately before pending-output ADD");
    require(f.graph->nodes[end - 1] == f.shared && f.shared->op == GGML_OP_MUL,
        "last prefix MUL is the ADD's shared branch");
    require(f.safe(why) && why == ov::Reject::none, "pinned disjoint synthetic allocations are safe");
    before.unchanged(f);
    require(f.cut(why) == end && f.safe(why), "cached cut can be revalidated on the same graph");
}

void variable_shapes_and_routing()
{
    struct Shape { int hidden, shared_ff, rows, used, experts; };
    // These are synthetic dimensions and independent routing capacities, not
    // model identities. Row batching here exercises metadata only; execution
    // policy can separately restrict which ordinary CPU path may overlap.
    for (const Shape shape : {Shape{4, 6, 1, 1, 1}, Shape{8, 12, 1, 2, 7},
                             Shape{32, 24, 2, 3, 13}, Shape{64, 96, 3, 5, 29},
                             Shape{16, 20, 1, 9, 61}, Shape{24, 40, 1, 17, 127}}) {
        Fixture f(shape.hidden, shape.shared_ff, shape.rows, shape.used, shape.experts);
        Snapshot before(f); ov::Reject why = ov::Reject::none;
        require(f.router->ne[0] == shape.experts && f.ids->ne[0] == shape.used &&
            f.ids->ne[1] == shape.rows && f.weights->ne[0] == shape.used,
            "routing metadata retains the requested expert capacity and actual selection shape");
        require(f.cut(why) == f.index(f.join) && f.safe(why),
            "independent shared-branch eligibility follows graph shape rather than model or expert-count constants");
        before.unchanged(f);
    }
}

template<class Change>
void reject_pending(Change change, const char* message)
{
    Fixture f(8, 12, 2, 3, 11); change(f); f.join->src[0] = f.pending;
    Snapshot before(f); ov::Reject why = ov::Reject::none;
    require(f.cut(why) < 0 && why == ov::Reject::geometry, message);
    require(!f.safe(why) && why == ov::Reject::geometry,
        "allocation validation independently rejects an invalid pending-result representation");
    before.unchanged(f);
}

void fresh_pending_contract()
{
    reject_pending([](Fixture& f) {
        auto* view = ggml_view_2d(f.ctx, f.pending, f.pending->ne[0], f.pending->ne[1], f.pending->nb[1], 0);
        // Deliberately malformed: ggml_set_input itself rejects non-leaves.
        view->flags |= GGML_TENSOR_FLAG_INPUT; ggml_set_output(view); f.pending = view;
    }, "even a zero-offset contiguous pending view is not a fresh result allocation");
    reject_pending([](Fixture& f) {
        auto* produced = ggml_cont(f.ctx, f.pending);
        produced->data = address(0x50000000);
        produced->flags |= GGML_TENSOR_FLAG_INPUT; ggml_set_output(produced); f.pending = produced;
    }, "a produced pending tensor cannot stand in for the host-written leaf");
    reject_pending([](Fixture& f) {
        auto* half = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F16, f.pending->ne[0], f.pending->ne[1]);
        half->data = address(0x50000000);
        ggml_set_input(half); ggml_set_output(half); f.pending = half;
    }, "pending-result storage must be F32");
    reject_pending([](Fixture& f) { f.pending->nb[1] += sizeof(float); },
        "a noncontiguous non-view pending root is rejected");
}

void allowed_metadata_prefix()
{
    Fixture f;
    auto* view = ggml_view_2d(f.ctx, f.x, 4, 1, f.x->nb[1], 0);
    auto* reshape = ggml_reshape_2d(f.ctx, view, 4, 1);
    auto* contiguous = ggml_cont(f.ctx, reshape);
    contiguous->data = address(0x60000000);
    std::move_backward(f.graph->nodes + 1, f.graph->nodes + f.graph->n_nodes,
        f.graph->nodes + f.graph->n_nodes + 3);
    f.graph->nodes[1] = view; f.graph->nodes[2] = reshape; f.graph->nodes[3] = contiguous;
    f.graph->n_nodes += 3; f.up->src[1] = contiguous;
    Snapshot before(f); ov::Reject why = ov::Reject::none;
    require(f.cut(why) == f.index(f.join) && f.safe(why),
        "VIEW and RESHAPE read aliases and an allocated CONT are allowed inside a valid prefix");
    before.unchanged(f);
}

template<class Change>
void reject_prefix(Change change, const char* message)
{
    Fixture f; change(f); ov::Reject why = ov::Reject::none;
    require(f.cut(why) < 0 && why != ov::Reject::none, message);
}

void structural_rejections()
{
    reject_prefix([](Fixture& f) { f.gate->src[1] = f.pending; }, "prefix cannot read pending output directly");
    reject_prefix([](Fixture& f) {
        auto* view = ggml_view_1d(f.ctx, f.pending, 2, sizeof(float));
        auto* nested = ggml_reshape_2d(f.ctx, view, 2, 1);
        nested->src[0] = nullptr; // view_src must independently expose the dependency.
        f.gate->src[1] = nested;
    }, "nested effective view roots cannot hide a pending-output dependency");
    reject_prefix([](Fixture& f) { f.up->op = GGML_OP_CPY; }, "CPY side effects are excluded");
    reject_prefix([](Fixture& f) { f.up->op = GGML_OP_SET; }, "SET side effects are excluded");
    reject_prefix([](Fixture& f) { f.gate->src[1] = f.join; }, "prefix cannot depend on a future rejoin producer");
    reject_prefix([](Fixture& f) { f.gate->src[1] = ggml_mul(f.ctx, f.x, f.x); },
        "a non-leaf source absent from the full graph fails closed");
    reject_prefix([](Fixture& f) {
        auto* later = ggml_mul(f.ctx, f.x, f.x);
        ggml_graph_add_node(f.graph, later);
        f.gate->src[1] = later;
    }, "future producer after the requested rejoin cannot enter a cached prefix");
    reject_prefix([](Fixture& f) { f.join->src[1] = f.down; }, "rejoin must consume the immediately preceding shared MUL");
    reject_prefix([](Fixture& f) { f.join->op = GGML_OP_MUL; }, "missing ADD rejoin fails closed");
    reject_prefix([](Fixture& f) { f.shared->op = GGML_OP_CONT; }, "cut cannot bisect the required MUL/ADD boundary");
    reject_prefix([](Fixture& f) {
        auto* non_shared = ggml_relu(f.ctx, f.gate);
        f.graph->nodes[f.index(f.activation->src[0])] = non_shared;
        f.activation->src[0] = non_shared;
    }, "unary operations outside SILU and SIGMOID are excluded");
    Fixture f; ov::Reject why = ov::Reject::none;
    require(ov::prefix_end(f.graph, -1, f.graph->n_nodes, f.pending, why) < 0, "negative begin is rejected");
    require(ov::prefix_end(f.graph, 0, f.graph->n_nodes + 1, f.pending, why) < 0, "limit beyond graph is rejected");
    require(ov::prefix_end(f.graph, f.index(f.join), f.graph->n_nodes, f.pending, why) < 0, "empty shared prefix is rejected");
    require(ov::prefix_end(f.graph, 0, f.index(f.join), f.pending, why) < 0, "rejoin outside the allowed interval is rejected");
    const int activation_begin = f.index(f.activation->src[0]);
    require(activation_begin > 0 &&
        ov::prefix_end(f.graph, activation_begin, f.graph->n_nodes, f.pending, why) < 0,
        "beginning at SILU cannot split the preceding matmul/activation fusion");

    Fixture oversized;
    for (int i = 0; i < 65; ++i)
        oversized.graph->nodes[i] = i == 0 ? oversized.gate : ggml_mul(oversized.ctx, oversized.x, oversized.x);
    oversized.join->src[1] = oversized.graph->nodes[64];
    oversized.graph->nodes[65] = oversized.join; oversized.graph->n_nodes = 66;
    require(oversized.cut(why) < 0, "shared prefix cannot exceed 64 nodes");
}

template<class Change>
void reject_allocation(Change change, ov::Reject expected, const char* message)
{
    Fixture f; change(f); ov::Reject why = ov::Reject::none;
    require(!f.safe(why) && why == expected, message);
}

void allocation_rejections()
{
    reject_allocation([](Fixture& f) { f.pending->flags &= ~GGML_TENSOR_FLAG_INPUT; }, ov::Reject::unpinned,
        "host-written pending output requires INPUT as well as OUTPUT");
    reject_allocation([](Fixture& f) { f.pending->flags &= ~GGML_TENSOR_FLAG_OUTPUT; }, ov::Reject::unpinned,
        "host-written pending output remains pinned after its last graph consumer");
    reject_allocation([](Fixture& f) { f.ids->flags &= ~GGML_TENSOR_FLAG_OUTPUT; }, ov::Reject::unpinned,
        "host boundary inputs must remain OUTPUT pinned");
    reject_allocation([](Fixture& f) {
        auto* root = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, 8);
        root->data = address(0x50000000);
        auto* view = ggml_view_1d(f.ctx, root, 4, 4 * sizeof(float));
        view->flags |= GGML_TENSOR_FLAG_OUTPUT; // Deliberately leave the storage root unpinned.
        f.inputs[0] = view;
    }, ov::Reject::unpinned, "a view OUTPUT bit does not substitute for a pinned allocation root");
    reject_allocation([](Fixture& f) { f.gate->data = nullptr; }, ov::Reject::allocation,
        "all prefix destinations require allocated data");
    reject_allocation([](Fixture& f) { f.gate->src[0]->data = nullptr; }, ov::Reject::allocation,
        "prefix weight sources require allocated data");
    reject_allocation([](Fixture& f) {
        f.gate->data = address(std::numeric_limits<std::uintptr_t>::max() - ggml_nbytes(f.gate) + 1);
    }, ov::Reject::allocation, "allocated byte ranges cannot wrap the address space");
    reject_allocation([](Fixture& f) { f.gate->data = f.pending->data; }, ov::Reject::alias,
        "distinct prefix destination and pending tensors cannot share storage");
    reject_allocation([](Fixture& f) { f.gate->src[0]->data = f.pending->data; }, ov::Reject::alias,
        "a prefix read cannot race the host pending-output write");
    reject_allocation([](Fixture& f) {
        f.gate->data = address(reinterpret_cast<std::uintptr_t>(f.x->data) + ggml_nbytes(f.x) - 1);
    }, ov::Reject::alias, "a one-byte write overlap with a protected input is unsafe");
    reject_allocation([](Fixture& f) {
        auto* view = ggml_view_1d(f.ctx, f.weights, 2, sizeof(float));
        ggml_set_output(view);
        f.inputs[2] = view;
        f.gate->data = view->data;
    }, ov::Reject::alias, "protected input views participate in effective allocation alias checks");
    reject_allocation([](Fixture& f) {
        auto* root = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, 128);
        root->data = address(0x70000000);
        auto* strided = ggml_view_2d(f.ctx, root, 1, 2, 64, 0);
        ggml_set_output(strided); f.inputs[2] = strided;
        f.gate->data = address(0x70000000 + 64);
    }, ov::Reject::alias, "protected strided views use their full byte span rather than element count");

    Fixture adjacent; ov::Reject why = ov::Reject::none;
    void* original_gate = adjacent.gate->data;
    adjacent.gate->data = address(reinterpret_cast<std::uintptr_t>(adjacent.x->data) + ggml_nbytes(adjacent.x));
    require(adjacent.safe(why), "half-open ranges permit an exactly adjacent destination");
    adjacent.gate->data = original_gate;
    adjacent.gate->src[0]->data = adjacent.x->data;
    require(adjacent.safe(why), "prefix and CPU reads may share protected input storage");
    const int cached_end = adjacent.index(adjacent.join);
    adjacent.up->data = adjacent.pending->data;
    require(!ov::safe_allocations(adjacent.graph, 0, cached_end, adjacent.pending,
        adjacent.inputs.data(), 3, why) && why == ov::Reject::alias,
        "cached node cuts still require allocation checks after rebinding");
}

struct FakeDrain {
    int calls = 0;
    bool result = true;
    static bool drain(void* state) noexcept
    {
        auto& fake = *static_cast<FakeDrain*>(state);
        ++fake.calls; return fake.result;
    }
};

void drain_lifetime()
{
    FakeDrain unused;
    { ov::Drain guard(FakeDrain::drain, &unused); require(guard.finish(), "unarmed finish succeeds"); }
    require(unused.calls == 0, "unarmed guards never drain");
    FakeDrain normal;
    {
        ov::Drain guard(FakeDrain::drain, &normal);
        guard.arm();
        require(normal.calls == 0, "arming precedes submission without draining");
        require(guard.finish() && guard.finish(), "explicit finish is idempotent");
    }
    require(normal.calls == 1, "normal submission drains exactly once including destruction");
    FakeDrain partial;
    { ov::Drain guard(FakeDrain::drain, &partial); guard.arm(); /* submission reports failure after queueing */ }
    require(partial.calls == 1, "failed submission still drains because it was armed first");
    FakeDrain failed; failed.result = false;
    {
        ov::Drain guard(FakeDrain::drain, &failed); guard.arm();
        require(!guard.finish(), "drain failure propagates");
        failed.result = true;
        require(!guard.finish(), "failure stays latched without a second drain");
        guard.arm();
        require(!guard.finish(), "a later successful drain cannot erase a prior failure");
    }
    require(failed.calls == 2, "each armed drain runs once even after failure");
    FakeDrain exception;
    try {
        ov::Drain guard(FakeDrain::drain, &exception); guard.arm();
        throw std::runtime_error("simulated host exception");
    } catch (const std::runtime_error&) {}
    require(exception.calls == 1, "exception unwinding drains outstanding work");
}
}

int main()
{
    try {
        eligible_prefix(); variable_shapes_and_routing(); fresh_pending_contract();
        allowed_metadata_prefix(); structural_rejections(); allocation_rejections(); drain_lifetime();
        std::printf("PASS %d model-free host MoE Metal-overlap metadata checks\n", checks);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL after %d checks: %s\n", checks, error.what());
        return 1;
    }
}
