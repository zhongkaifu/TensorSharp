// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <map>
#include <memory>
#include <utility>

namespace tsg {
// The identity is the managed weight's stable host pointer, not a transient
// ggml tensor address or its backend-specific resident address.
ggml_tensor * bonsai_mul_mat(ggml_context *, ggml_tensor *, ggml_tensor *, const void *);
ggml_tensor * bonsai_get_rows(ggml_context *, ggml_tensor *, ggml_tensor *, const void *);
ggml_tensor * bonsai_transform(ggml_context *, ggml_tensor *, const void *, bool inverse);
bool bonsai_weight_has_transform(const void *);
// Whether any weight carries a transform (a cheap test before looking one up).
bool bonsai_any_transform();
void bonsai_clear_backend();

// One rotated activation per (input, transform) for a graph built over many calls (the graph builder's), where a
// BonsaiGraphScope cannot span them: Q/K/V and gate/up products of one input share it.
struct BonsaiMemo {
    std::map<std::pair<const ggml_tensor *, const void *>, ggml_tensor *> entries;
};
// w · x, with x rotated first when `key` names a weight with a forward transform (else a plain product).
ggml_tensor * bonsai_mul_mat_memo(ggml_context *, ggml_tensor * w, ggml_tensor * x, const void * key, BonsaiMemo &);

// Share transformed activations between Q/K/V and gate/up within one graph.
// A scoped cache cannot accidentally reuse tensors from a freed ggml context.
struct BonsaiGraphScope {
    struct Impl;
    std::unique_ptr<Impl> impl;
    BonsaiGraphScope();
    ~BonsaiGraphScope();
    BonsaiGraphScope(const BonsaiGraphScope &) = delete;
    BonsaiGraphScope & operator=(const BonsaiGraphScope &) = delete;
};
}
