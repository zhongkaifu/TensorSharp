// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>

namespace tsg {

// One resident weight to upload. `data` is where the caller's mapping holds its
// bytes; `done` is set when the upload read them from the file instead.
struct H3DirectReadItem
{
    ggml_tensor* tensor;
    const void* data;
    std::size_t bytes;
    bool done;
};

// Uploads every item whose bytes lie in a mapping registered with
// TSGgml_DirectReadRegister by reading the file itself, and returns how many it
// uploaded. The others, and all of them when it returns 0, are left to the caller:
// see ggml_ops_direct_read.cpp.
std::size_t h3_direct_read_upload(ggml_backend_t backend, H3DirectReadItem* items, std::size_t count,
                                  const char* tag);

} // namespace tsg
