// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_moe_pipeline_timeline.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-cuda/common.cuh"
namespace tsg {
static void CUDART_CB record_pipeline_clock(void* pointer) noexcept {
    static_cast<std::atomic<std::int64_t>*>(pointer)->store(moe_pipeline_clock_ns(), std::memory_order_release);
}
bool moe_pipeline_stream_mark(ggml_backend_t backend, std::atomic<std::int64_t>* target) {
    if (!backend || !target || !ggml_backend_is_cuda(backend)) return false;
    auto* context = static_cast<ggml_backend_cuda_context*>(backend->context);
    if (cudaSetDevice(context->device) != cudaSuccess) return false;
    return cudaLaunchHostFunc(context->stream(), record_pipeline_clock, target) == cudaSuccess;
}
}
