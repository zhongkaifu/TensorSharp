// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "ggml.h"

// Prefill-only weighted expert reduction. Returns null for decode/verify rows
// and unsupported layouts, so callers retain their existing graph. CUDA callers
// must execute the returned CUSTOM node through the TensorSharp fused backend.
// Inputs are F32 experts [width, used, tokens] and routes [1, used, tokens];
// row/token strides may include padding or a tensor-parallel output slice.
ggml_tensor * tsg_q4e_prefill_expert_reduce(ggml_context * ctx,
    ggml_tensor * experts, ggml_tensor * routes);
