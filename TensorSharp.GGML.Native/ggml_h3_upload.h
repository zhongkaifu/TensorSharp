// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3's weight uploader over a set of bindings: direct reads from a registered
// file mapping where they apply (ggml_h3_direct_read.h), else prefaulted, parallel
// copies out of the mapping (ggml_h3_weight_upload.h). MiniMax-H3's native graphs
// upload their resident weights with it, and so does the graph builder's weight
// preload (TSGgml_GraphPreloadWeights). Moved verbatim from ggml_ops_minimax_h3.cpp.
#pragma once

#include "ggml_ops_internal.h"
#include "ggml_h3_weight_upload.h"
#include "ggml_h3_direct_read.h"

#include <cstddef>
#include <vector>

namespace tsg {

// Uploads one entry's resident weights. Those bound into a mapping registered for
// direct reads come straight from the file (ggml_ops_direct_read.cpp); the rest,
// which is all of them where that is unavailable, are copied out of the mapping.
template <typename Bindings>
inline void h3_upload_weights(const char* tag, const Bindings& uploads)
{
    std::vector<H3DirectReadItem> items;
    items.reserve(uploads.size());
    for (const auto& u : uploads) items.push_back({ u.t, u.data, u.bytes, false });
    h3_direct_read_upload(g_backend, items.data(), items.size(), tag);

    struct Mapped { ggml_tensor* t; const void* data; std::size_t bytes; };
    std::vector<Mapped> mapped;
    for (const auto& item : items)
        if (!item.done) mapped.push_back({ item.tensor, item.data, item.bytes });
    H3WeightUpload weight_upload(g_backend, tag);
    weight_upload.prefault(mapped);
    for (const auto& m : mapped) weight_upload.set(m.t, m.data, m.bytes);
    weight_upload.finish();
}

} // namespace tsg
