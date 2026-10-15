// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"

namespace tsg {
// Model-owned stable weight keys select precision without changing other models.
ggml_tensor* weight_mul_mat(ggml_context*, ggml_tensor*, ggml_tensor*, const void* key);
ggml_backend_t q8_f32_execution_backend(ggml_backend_t, ggml_cgraph*);
void clear_q8_f32_backends();
}
