// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "dsv4_host_expert_read.h"
#include "ggml.h"

#if defined(__linux__)
namespace tsg_dsv4 {
struct host_expert_read_context {
    host_expert_reader * reader = nullptr;
    std::array<file_warm_range, 3> projections;
    size_t experts = 0;
    expert_residency_hints hints;
};

inline void host_expert_read_selected(host_expert_read_context & context, const ggml_tensor * src) noexcept {
    try {
        if (!src || src->type != GGML_TYPE_I32 || src->ne[2] != 1 || src->ne[3] != 1)
            throw std::runtime_error("Host expert read: expected a 2D I32 route tensor");
        std::vector<int32_t> ids;
        ids.reserve(size_t(ggml_nelements(src)));
        for (int64_t j = 0; j < src->ne[1]; ++j) for (int64_t i = 0; i < src->ne[0]; ++i) {
            int32_t id;
            memcpy(&id, static_cast<const char *>(src->data) + i * src->nb[0] + j * src->nb[1], sizeof(id));
            ids.push_back(id);
        }
        const auto needed = context.hints.select(ids, context.experts, src->ne[1] == 1);
        if (!needed.empty()) {
            context.reader->warm(selected_expert_ranges(context.projections, context.experts, needed));
            if (!context.reader->error()[0]) context.hints.mark(needed);
        }
    } catch (const std::exception & ex) { context.reader->fail(ex.what()); }
    catch (...) { context.reader->fail("Host expert read: route preparation failed"); }
}

} // namespace tsg_dsv4
#endif
