// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml-backend-impl.h"

namespace tsg_glm {

// A scheduler allocation descriptor is not a vendor buffer type: CUDA and
// other devices may recognize their types by interface/pointer identity. This
// graph-local device facade validates the original type before accepting the
// descriptor. Physical buffers and the underlying backend remain untouched.
class graph_backend {
public:
    using unwrap_type = ggml_backend_buffer_type_t (*)(void*, ggml_backend_buffer_type_t);
private:
    ggml_backend_t original_;
    void* owner_;
    unwrap_type unwrap_;
    ggml_backend_device device_{};
    ggml_backend backend_{};
    static graph_backend& self(ggml_backend_dev_t device) { return *static_cast<graph_backend*>(device->context); }
    ggml_backend_dev_t original_device() const { return original_->device; }
    static const char* name(ggml_backend_dev_t d) { return ggml_backend_dev_name(self(d).original_device()); }
    static const char* description(ggml_backend_dev_t d) { return ggml_backend_dev_description(self(d).original_device()); }
    static void memory(ggml_backend_dev_t d, size_t* free, size_t* total) { ggml_backend_dev_memory(self(d).original_device(), free, total); }
    static enum ggml_backend_dev_type type(ggml_backend_dev_t d) { return ggml_backend_dev_type(self(d).original_device()); }
    static void props(ggml_backend_dev_t d, struct ggml_backend_dev_props* p) { ggml_backend_dev_get_props(self(d).original_device(), p); }
    static ggml_backend_t init(ggml_backend_dev_t d, const char* params) { return ggml_backend_dev_init(self(d).original_device(), params); }
    static ggml_backend_buffer_type_t buffer_type(ggml_backend_dev_t d) { return ggml_backend_dev_buffer_type(self(d).original_device()); }
    static ggml_backend_buffer_type_t host_buffer_type(ggml_backend_dev_t d) { return ggml_backend_dev_host_buffer_type(self(d).original_device()); }
    static ggml_backend_buffer_t from_host(ggml_backend_dev_t d, void* p, size_t bytes, size_t max_tensor) {
        return ggml_backend_dev_buffer_from_host_ptr(self(d).original_device(), p, bytes, max_tensor);
    }
    static bool supports_op(ggml_backend_dev_t d, const ggml_tensor* op) { return ggml_backend_dev_supports_op(self(d).original_device(), op); }
    static bool supports_type(ggml_backend_dev_t d, ggml_backend_buffer_type_t t) {
        auto& facade = self(d);
        return ggml_backend_dev_supports_buft(facade.original_device(), facade.unwrap_(facade.owner_, t));
    }
    static bool offload(ggml_backend_dev_t d, const ggml_tensor* op) { return ggml_backend_dev_offload_op(self(d).original_device(), op); }
    static ggml_backend_event_t event_new(ggml_backend_dev_t d) { return ggml_backend_event_new(self(d).original_device()); }
    static void event_free(ggml_backend_dev_t, ggml_backend_event_t event) { ggml_backend_event_free(event); }
    static void event_sync(ggml_backend_dev_t, ggml_backend_event_t event) { ggml_backend_event_synchronize(event); }
    static graph_backend& backend_self(ggml_backend_t b) { return self(b->device); }
    static ggml_backend_t unwrap_backend(ggml_backend_t b) {
        return b && b->iface.get_name == backend_name ? backend_self(b).original_ : b;
    }
    static const char* backend_name(ggml_backend_t b) { return ggml_backend_name(backend_self(b).original_); }
    static void free_alias(ggml_backend_t) {} // the model, not this graph, owns the stream
    static void set_tensor(ggml_backend_t b, ggml_tensor* t, const void* p, size_t off, size_t bytes) {
        auto original = backend_self(b).original_; original->iface.set_tensor_async(original, t, p, off, bytes);
    }
    static void get_tensor(ggml_backend_t b, const ggml_tensor* t, void* p, size_t off, size_t bytes) {
        auto original = backend_self(b).original_; original->iface.get_tensor_async(original, t, p, off, bytes);
    }
    static void set_tensor_2d(ggml_backend_t b, ggml_tensor* t, const void* p, size_t off, size_t bytes,
        size_t n, size_t stride_tensor, size_t stride_data) {
        auto original = backend_self(b).original_;
        original->iface.set_tensor_2d_async(original, t, p, off, bytes, n, stride_tensor, stride_data);
    }
    static void get_tensor_2d(ggml_backend_t b, const ggml_tensor* t, void* p, size_t off, size_t bytes,
        size_t n, size_t stride_tensor, size_t stride_data) {
        auto original = backend_self(b).original_;
        original->iface.get_tensor_2d_async(original, t, p, off, bytes, n, stride_tensor, stride_data);
    }
    static bool copy_tensor(ggml_backend_t src_backend, ggml_backend_t dst_backend,
        const ggml_tensor* src, ggml_tensor* dst) {
        src_backend = unwrap_backend(src_backend); dst_backend = unwrap_backend(dst_backend);
        return dst_backend->iface.cpy_tensor_async(src_backend, dst_backend, src, dst);
    }
    static void synchronize(ggml_backend_t b) { ggml_backend_synchronize(backend_self(b).original_); }
    static ggml_backend_graph_plan_t plan_create(ggml_backend_t b, const ggml_cgraph* graph) {
        auto original = backend_self(b).original_; return original->iface.graph_plan_create(original, graph);
    }
    static void plan_free(ggml_backend_t b, ggml_backend_graph_plan_t plan) {
        auto original = backend_self(b).original_; original->iface.graph_plan_free(original, plan);
    }
    static void plan_update(ggml_backend_t b, ggml_backend_graph_plan_t plan, const ggml_cgraph* graph) {
        auto original = backend_self(b).original_; original->iface.graph_plan_update(original, plan, graph);
    }
    static ggml_status plan_compute(ggml_backend_t b, ggml_backend_graph_plan_t plan) {
        auto original = backend_self(b).original_; return original->iface.graph_plan_compute(original, plan);
    }
    static ggml_status graph_compute(ggml_backend_t b, ggml_cgraph* graph) {
        auto original = backend_self(b).original_; return original->iface.graph_compute(original, graph);
    }
    static void record_event(ggml_backend_t b, ggml_backend_event_t event) {
        auto original = backend_self(b).original_; original->iface.event_record(original, event);
    }
    static void wait_event(ggml_backend_t b, ggml_backend_event_t event) {
        auto original = backend_self(b).original_; original->iface.event_wait(original, event);
    }
    static void optimize(ggml_backend_t b, ggml_cgraph* graph, ggml_backend_graph_optimize_params* params) {
        auto original = backend_self(b).original_; original->iface.graph_optimize(original, graph, params);
    }
public:
    graph_backend(ggml_backend_t original, void* owner, unwrap_type unwrap)
        : original_(original), owner_(owner), unwrap_(unwrap) {
        const auto device = original->device;
        device_.iface = { name, description, memory, type, props, init, buffer_type,
            device->iface.get_host_buffer_type ? host_buffer_type : nullptr,
            device->iface.buffer_from_host_ptr ? from_host : nullptr,
            supports_op, supports_type, device->iface.offload_op ? offload : nullptr,
            device->iface.event_new ? event_new : nullptr,
            device->iface.event_free ? event_free : nullptr,
            device->iface.event_synchronize ? event_sync : nullptr };
        device_.reg = device->reg;
        device_.context = this;
        // Preserve vendor context and GUID, including ggml_backend_is_cuda.
        // Every dispatch wrapper forwards the original backend pointer, while
        // direct vendor extension casts still see its original stream context.
        backend_ = *original;
        backend_.device = &device_;
        const auto& iface = original->iface;
        backend_.iface = { backend_name, free_alias,
            iface.set_tensor_async ? set_tensor : nullptr, iface.get_tensor_async ? get_tensor : nullptr,
            iface.set_tensor_2d_async ? set_tensor_2d : nullptr, iface.get_tensor_2d_async ? get_tensor_2d : nullptr,
            iface.cpy_tensor_async ? copy_tensor : nullptr, iface.synchronize ? synchronize : nullptr,
            iface.graph_plan_create ? plan_create : nullptr, iface.graph_plan_free ? plan_free : nullptr,
            iface.graph_plan_update ? plan_update : nullptr, iface.graph_plan_compute ? plan_compute : nullptr,
            graph_compute, iface.event_record ? record_event : nullptr, iface.event_wait ? wait_event : nullptr,
            iface.graph_optimize ? optimize : nullptr };
    }
    graph_backend(const graph_backend&) = delete;
    graph_backend& operator=(const graph_backend&) = delete;
    ggml_backend_t get() { return &backend_; }
    ggml_backend_t original() const { return original_; }
};
}
