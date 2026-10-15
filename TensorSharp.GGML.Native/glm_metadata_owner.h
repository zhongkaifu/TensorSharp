// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include "ggml.h"
#include "gguf.h"

namespace tsg_glm {

// gguf_init_from_file publishes two independent owners. Keep both alive while
// copying metadata; source-map/string allocation may throw before explicit
// cleanup. Release the GGUF context before its accompanying tensor context,
// matching the loader's original successful-path order.
template<void (*FreeFile)(gguf_context *) = gguf_free,
         void (*FreeTensors)(ggml_context *) = ggml_free>
struct metadata_owner {
    ggml_context * tensors = nullptr;
    gguf_context * file = nullptr;

    metadata_owner() = default;
    metadata_owner(const metadata_owner &) = delete;
    metadata_owner & operator=(const metadata_owner &) = delete;
    ~metadata_owner() { reset(); }
    void reset() noexcept {
        if (file) { FreeFile(file); file = nullptr; }
        if (tensors) { FreeTensors(tensors); tensors = nullptr; }
    }
};

} // namespace tsg_glm
