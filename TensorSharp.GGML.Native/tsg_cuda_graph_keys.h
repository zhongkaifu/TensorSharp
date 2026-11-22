// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
#pragma once

#include "ggml-backend.h"

// Switch ggml-cuda's own CUDA graph capture off for the node runs whose first nodes are `keys` (tsg_cuda_graph_keys.cu):
// for a caller that captures those runs itself.
void tsg_cuda_disable_graphs(ggml_backend_t cuda_backend, const void * const * keys, int count);

// Whether ggml-cuda runs every node of `graph` without synchronizing its stream, so a stream capture can record it:
// ggml-cuda's own rule (no MUL_MAT_ID on the fallback that sorts the expert ids on the host).
bool tsg_cuda_graph_capturable(ggml_backend_t cuda_backend, const struct ggml_cgraph * graph);
