// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_matmul_precision.h"
#include "ggml_ops_dsv4_fused.h"
#include "ggml_ops_precision_policy.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cuda.h"
#include "ggml-cuda/common.cuh"
#include <algorithm>
#include <climits>
#include <cstring>
#include <stdexcept>
#include <cstdlib>
#include <string>
#include "ggml_ops_matmul_quant_strip.cuh"

namespace {
struct tensor_layout {
    int64_t ne[4];
    size_t nb[4];
};
tensor_layout layout(const ggml_tensor * t) {
    tensor_layout l;
    for (int d = 0; d < 4; ++d) { l.ne[d] = t->ne[d]; l.nb[d] = t->nb[d]; }
    return l;
}
__device__ __forceinline__ float load(const char * p, int type) {
    if (type == GGML_TYPE_F32) return *reinterpret_cast<const float *>(p);
    if (type == GGML_TYPE_F16) return __half2float(*reinterpret_cast<const half *>(p));
    return __uint_as_float(uint32_t(*reinterpret_cast<const uint16_t *>(p)) << 16);
}
__global__ void pack_f32(const char * source, float * destination, tensor_layout l, int type, int64_t count) {
    for (int64_t n = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; n < count;
         n += int64_t(blockDim.x) * gridDim.x) {
        int64_t rest = n;
        size_t offset = 0;
        for (int d = 0; d < 4; ++d) { offset += (rest % l.ne[d]) * l.nb[d]; rest /= l.ne[d]; }
        destination[n] = load(source + offset, type);
    }
}

// One warp per output row/selected token. Activations stay F32; F16/BF16
// weights widen exactly before FMA. Expert IDs remain on device, avoiding a
// synchronization or a dense expansion of every expert during decode.
__global__ void indexed_f32(const char * weights, const char * input, const char * ids, float * output,
        tensor_layout a, tensor_layout b, tensor_layout selections, int type, int64_t rows, int64_t used,
        int64_t count) {
    const int lane = threadIdx.x & 31;
    const int64_t first = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const int64_t step = int64_t(blockDim.x) * gridDim.x / 32;
    for (int64_t index = first; index < count; index += step) {
        const int64_t row = index % rows;
        const int64_t slot = (index / rows) % used;
        const int64_t token = index / (rows * used);
        const int expert = *reinterpret_cast<const int32_t *>(ids + slot * selections.nb[0] + token * selections.nb[1]);
        if (expert < 0 || expert >= a.ne[2]) {
            if (lane == 0) output[index] = nanf("");
            continue;
        }
        const char * aw = weights + row * a.nb[1] + expert * a.nb[2];
        const char * bx = input + (slot % b.ne[1]) * b.nb[1] + token * b.nb[2];
        float sum = 0;
        // Adjacent pairs both improve memory locality and avoid separating
        // alternating positive/negative terms into huge same-sign partials.
        for (int64_t k = 2 * lane; k < a.ne[0]; k += 64) {
            sum = fmaf(load(aw + k * a.nb[0], type), load(bx + k * b.nb[0], GGML_TYPE_F32), sum);
            if (k + 1 < a.ne[0])
                sum = fmaf(load(aw + (k + 1) * a.nb[0], type), load(bx + (k + 1) * b.nb[0], GGML_TYPE_F32), sum);
        }
        for (int shift = 16; shift > 0; shift /= 2) sum += __shfl_down_sync(0xffffffff, sum, shift);
        if (lane == 0) output[index] = sum;
    }
}

// Decode has only a few columns. A warp reduction avoids cuBLAS submission
// overhead and avoids allocating a widened copy of an F16 attention cache.
__global__ void vector_f32(const char * weights, const char * input, float * output,
        tensor_layout a, tensor_layout b, int type, int64_t count) {
    const int lane = threadIdx.x & 31;
    const int64_t first = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const int64_t step = int64_t(blockDim.x) * gridDim.x / 32;
    for (int64_t index = first; index < count; index += step) {
        int64_t rest = index;
        const int64_t row = rest % a.ne[1]; rest /= a.ne[1];
        const int64_t column = rest % b.ne[1]; rest /= b.ne[1];
        const int64_t i2 = rest % b.ne[2], i3 = rest / b.ne[2];
        const char * aw = weights + row * a.nb[1] + (i2 / (b.ne[2] / a.ne[2])) * a.nb[2]
            + (i3 / (b.ne[3] / a.ne[3])) * a.nb[3];
        const char * bx = input + column * b.nb[1] + i2 * b.nb[2] + i3 * b.nb[3];
        float sum = 0;
        for (int64_t k = 2 * lane; k < a.ne[0]; k += 64) {
            sum = fmaf(load(aw + k * a.nb[0], type), load(bx + k * b.nb[0], GGML_TYPE_F32), sum);
            if (k + 1 < a.ne[0])
                sum = fmaf(load(aw + (k + 1) * a.nb[0], type), load(bx + (k + 1) * b.nb[0], GGML_TYPE_F32), sum);
        }
        for (int shift = 16; shift > 0; shift /= 2) sum += __shfl_down_sync(0xffffffff, sum, shift);
        if (lane == 0) output[index] = sum;
    }
}
} // namespace

struct tsg_matmul_cuda_state {
    int device;
    cudaStream_t stream;
    cublasHandle_t handle = nullptr;
    float * scratch = nullptr;
    size_t capacity = 0;
};

namespace {
void reserve_scratch(tsg_matmul_cuda_state * state, size_t needed) {
    if (needed <= state->capacity) return;
    // Fused nodes run between captured graph views. Scratch belongs to this
    // TensorSharp backend and persists across nodes; no ggml private pool or
    // allocation-layout assumptions are involved. Growth waits for prior users.
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(state->stream, &capture));
    GGML_ASSERT(capture == cudaStreamCaptureStatusNone);
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    size_t requested = needed;
#if defined(TSG_GGML_TEST_HOOKS)
    if (const char * fault = std::getenv("TS_TP_TEST_FAIL_SCRATCH_GROWTH"))
        if (fault[0] == '1') throw std::runtime_error("Injected TensorSharp CUDA matmul scratch growth failure");
    if (const char * fault = std::getenv("TS_TP_TEST_EXHAUST_SCRATCH")) {
        if (fault[0] == '1') {
            size_t available, total;
            CUDA_CHECK(cudaMemGetInfo(&available, &total));
            // A bounded fixture forces a real allocation rejection without
            // consuming another request's available device memory.
            GGML_ASSERT(total <= SIZE_MAX / 2);
            requested = total * 2;
        }
    }
#endif
    float * replacement = nullptr;
    const auto status = cudaMalloc(reinterpret_cast<void **>(&replacement), requested);
    if (status != cudaSuccess) {
        // This synchronous allocation failure is handled by the model's
        // exception boundary. Leave the old allocation usable on retry and
        // do not let the handled error poison the next kernel's error check.
        cudaGetLastError();
        throw std::runtime_error(std::string("Cannot grow TensorSharp CUDA matmul scratch: ") + cudaGetErrorString(status));
    }
    if (state->scratch) CUDA_CHECK(cudaFree(state->scratch));
    state->scratch = replacement;
    state->capacity = needed;
}

void compute_quant_strip(tsg_matmul_cuda_state * state, ggml_tensor * dst) {
    ggml_custom_op_params params;
    std::memcpy(&params, dst->op_params, sizeof(params));
    const auto * desc = static_cast<const tsg_dsv4_fused_desc *>(params.userdata);
    const auto * w = dst->src[0], * x = dst->src[1], * ids = dst->src[2];
    const int tokens = int(x->ne[2]), used = int(ids->ne[0]), selected_rows = tokens * used;
    const int width = quant_strip_width(w->type, tokens, state->device);
    const int cc = ggml_cuda_info().devices[state->device].cc;
    const auto config = ggml_cuda_mmq_get_config(w->type, width, false, cc);
    const int padded = GGML_PAD(x->ne[0], MATRIX_ROW_PADDING);
    const auto align = [](size_t n) { return (n + 255) / 256 * 256; };
    const size_t input_offset = 0;
    const size_t output_offset = align(size_t(selected_rows) * sizeof(int32_t));
    const size_t bounds_offset = output_offset * 2;
    const size_t quant_offset = bounds_offset + align(size_t(w->ne[2] + 1) * sizeof(int32_t));
    const size_t quant_bytes = size_t(selected_rows) * padded * sizeof(block_q8_1_mmq) / QK8_1_MMQ
        + ggml_cuda_mmq_get_J_max(w->type, false, cc, 1) * sizeof(block_q8_1_mmq);
    const size_t partial_offset = quant_offset + align(quant_bytes);
    quant_strip_args a{static_cast<const char *>(w->data), nullptr, nullptr, nullptr,
        static_cast<float *>(dst->data), nullptr, desc->i0, desc->i1, int(w->ne[1]),
        int(w->nb[1] / ggml_type_size(w->type)), int(w->nb[2] / ggml_type_size(w->type)),
        selected_rows, tokens, int(w->ne[2]), init_fastdiv_values(w->ne[0] / ggml_blck_size(w->type)),
        init_fastdiv_values((tokens + width - 1) / width)};
    const int blocks = quant_strip_active_blocks(a, config, state->device);
    const int matrices = dst->src[3] ? 2 : 1;
    const size_t partial_bytes = size_t(blocks) * width * config.I * sizeof(float);
    reserve_scratch(state, partial_offset + matrices * partial_bytes);
    auto * scratch = reinterpret_cast<char *>(state->scratch);
    auto * input_ids = reinterpret_cast<int32_t *>(scratch + input_offset);
    auto * output_ids = reinterpret_cast<int32_t *>(scratch + output_offset);
    auto * bounds = reinterpret_cast<int32_t *>(scratch + bounds_offset);
    const bool dedup = used > 1;
    ggml_cuda_launch_mm_ids_helper(static_cast<const int32_t *>(ids->data), input_ids, output_ids, bounds,
        w->ne[2], tokens, used, 1, ids->nb[1] / sizeof(int32_t), x->nb[2] / x->nb[1], dedup, state->stream);
    CUDA_CHECK(cudaGetLastError());
    if (dedup)
        quantize_scatter_mmq_q8_1_cuda(static_cast<const float *>(x->data), input_ids, scratch + quant_offset,
            w->type, x->ne[0], x->nb[2] / sizeof(float), padded, tokens, selected_rows, used, state->stream);
    else
        quantize_mmq_q8_1_cuda(static_cast<const float *>(x->data), input_ids, scratch + quant_offset,
            w->type, x->ne[0], x->nb[1] / sizeof(float), x->nb[2] / sizeof(float), x->nb[3] / sizeof(float),
            padded, selected_rows, 1, 1, state->stream);
    CUDA_CHECK(cudaGetLastError());
    a.input = reinterpret_cast<const int *>(scratch + quant_offset);
    a.ids = output_ids;
    a.bounds = bounds;
    a.partial = reinterpret_cast<float *>(scratch + partial_offset);
    if (matrices == 2) {
        a.weights2 = static_cast<const char *>(dst->src[3]->data);
        a.out2 = reinterpret_cast<float *>(static_cast<char *>(dst->data)+dst->nb[3]);
        a.partial2 = reinterpret_cast<float *>(scratch + partial_offset + partial_bytes);
    }
    if (w->type == GGML_TYPE_Q2_K) dispatch_quant_strip<GGML_TYPE_Q2_K>(state->device, state->stream, width, a);
    else dispatch_quant_strip<GGML_TYPE_Q4_K>(state->device, state->stream, width, a);
}
} // namespace

bool tsg_matmul_id_quant_strip_supported(ggml_backend_t backend, const ggml_tensor * w,
        int64_t tokens, int64_t full_rows, int64_t first_row) {
    if (!ggml_backend_is_cuda(backend) || (w->type != GGML_TYPE_Q2_K && w->type != GGML_TYPE_Q4_K) ||
        !ggml_is_contiguous(w) || tokens <= 0 || tokens > INT_MAX || w->ne[0] > INT_MAX ||
        full_rows <= 0 || full_rows > INT_MAX || first_row < 0 || first_row + w->ne[1] > full_rows ||
        w->ne[2] > INT_MAX / tokens || w->ne[0] > INT_MAX - MATRIX_ROW_PADDING ||
        w->nb[2] / ggml_type_size(w->type) > INT_MAX) return false;
    const int device = static_cast<ggml_backend_cuda_context *>(backend->context)->device;
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || tokens <= get_mmvq_mmid_max_batch(w->type, cc) ||
        !ggml_cuda_should_use_mmq(w->type, cc, tokens, w->ne[2])) return false;
    const int width = quant_strip_width(w->type, int(tokens), device);
    if (!width) return false;
    const auto config = ggml_cuda_mmq_get_config(w->type, width, false, cc);
    const int64_t tiles = (full_rows / config.I) * ((tokens + width - 1) / width) * w->ne[2];
    // Nonaligned layouts and non-stream-K architectures remain on the
    // existing upstream route; those configurations need separate qualification.
    return config.stream_k && tiles <= INT_MAX / (w->ne[0] / ggml_blck_size(w->type)) &&
        full_rows % config.I == 0 && first_row % config.I == 0 &&
        w->ne[1] % config.I == 0 && w->ne[1] % 128 == 0 && full_rows % 128 == 0;
}

tsg_matmul_cuda_state * tsg_matmul_cuda_init(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_cuda(backend));
    auto * context = static_cast<ggml_backend_cuda_context *>(backend->context);
    CUDA_CHECK(cudaSetDevice(context->device));
    auto * state = new tsg_matmul_cuda_state;
    state->device = context->device;
    state->stream = context->stream(context->device, 0);
    // A private handle prevents ggml's TF32 settings and concurrent ranks from
    // altering this stream or math mode. Creation happens before execution.
    CUBLAS_CHECK(cublasCreate(&state->handle));
    CUBLAS_CHECK(cublasSetStream(state->handle, state->stream));
    CUBLAS_CHECK(cublasSetMathMode(state->handle, CUBLAS_PEDANTIC_MATH));
    return state;
}

void tsg_matmul_cuda_free(tsg_matmul_cuda_state * state) {
    if (!state) return;
    CUDA_CHECK(cudaSetDevice(state->device));
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    if (state->scratch) CUDA_CHECK(cudaFree(state->scratch));
    CUBLAS_CHECK(cublasDestroy(state->handle));
    delete state;
}

void tsg_matmul_cuda_compute(tsg_matmul_cuda_state * state, ggml_tensor * dst) {
    const auto * a = dst->src[0];
    const auto * b = dst->src[1];
    GGML_ASSERT(ggml_is_contiguous(dst));
    CUDA_CHECK(cudaSetDevice(state->device));
    if (a->type == GGML_TYPE_Q2_K || a->type == GGML_TYPE_Q4_K) {
        compute_quant_strip(state, dst);
        return;
    }
    if (dst->src[2]) {
        const int64_t count = ggml_nelements(dst);
        const unsigned blocks = unsigned(std::min<int64_t>((count + 3) / 4, 65535));
        indexed_f32<<<blocks, 128, 0, state->stream>>>(static_cast<const char *>(a->data),
            static_cast<const char *>(b->data), static_cast<const char *>(dst->src[2]->data),
            static_cast<float *>(dst->data), layout(a), layout(b), layout(dst->src[2]),
            a->type, dst->ne[0], dst->ne[1], count);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // Decode-class widths take the warp kernel: one warp per output element,
    // so a column's reduction order never depends on how many columns share
    // the launch. cuBLAS below picks its kernel by (m, n, k) and does not
    // promise that, and V4.1 quantizes this output into its caches, so a
    // speculative verify (block_size + 1 columns) has to reduce exactly like
    // the single-token decode it stands in for (see the policy header).
    if (b->ne[1] <= TSG_PRECISION_DECODE_COLUMNS) {
        const int64_t count = ggml_nelements(dst);
        const unsigned blocks = unsigned(std::min<int64_t>((count + 3) / 4, 65535));
        vector_f32<<<blocks, 128, 0, state->stream>>>(static_cast<const char *>(a->data),
            static_cast<const char *>(b->data), static_cast<float *>(dst->data),
            layout(a), layout(b), a->type, count);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const bool pack_a = a->type != GGML_TYPE_F32 || a->nb[0] != sizeof(float) || a->nb[1] % sizeof(float);
    const bool pack_b = b->nb[0] != sizeof(float) || b->nb[1] % sizeof(float);
    const size_t count_a = pack_a ? (size_t(ggml_nelements(a)) + 63) / 64 * 64 : 0;
    const size_t count_b = pack_b ? size_t(ggml_nelements(b)) : 0;
    const size_t needed = (count_a + count_b) * sizeof(float);
    reserve_scratch(state, needed);
    auto prepare = [&](const ggml_tensor * t, bool pack, float * scratch, tensor_layout & l) -> const char * {
        l = layout(t);
        if (!pack) return static_cast<const char *>(t->data);
        const int64_t count = ggml_nelements(t);
        const unsigned blocks = unsigned(std::min<int64_t>((count + 255) / 256, 65535));
        pack_f32<<<blocks, 256, 0, state->stream>>>(static_cast<const char *>(t->data), scratch, l, t->type, count);
        CUDA_CHECK(cudaGetLastError());
        l.nb[0] = sizeof(float);
        for (int d = 1; d < 4; ++d) l.nb[d] = l.nb[d - 1] * l.ne[d - 1];
        return reinterpret_cast<const char *>(scratch);
    };
    tensor_layout al, bl;
    const char * ap = prepare(a, pack_a, state->scratch, al);
    const char * bp = prepare(b, pack_b, state->scratch ? state->scratch + count_a : nullptr, bl);
    const float alpha = 1.0f, beta = 0.0f;
    GGML_ASSERT(a->ne[0] <= INT_MAX && a->ne[1] <= INT_MAX && b->ne[1] <= INT_MAX);
    GGML_ASSERT(al.nb[1] / sizeof(float) <= INT_MAX && bl.nb[1] / sizeof(float) <= INT_MAX);
    const int m = int(a->ne[1]), n = int(b->ne[1]), k = int(a->ne[0]);
    auto gemm = [&](const char * aw, const char * bx, float * output,
                    size_t stride_a, size_t stride_b, int batch) {
        // Pedantic compute prohibits TF32 and input narrowing regardless of
        // NVIDIA_TF32_OVERRIDE or GGML_CUDA_CUBLAS_COMPUTE_TYPE settings.
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(state->handle, CUBLAS_OP_T, CUBLAS_OP_N,
            m, n, k, &alpha, aw, CUDA_R_32F, int(al.nb[1] / sizeof(float)), stride_a / sizeof(float),
            bx, CUDA_R_32F, int(bl.nb[1] / sizeof(float)), stride_b / sizeof(float),
            &beta, output, CUDA_R_32F, m, int64_t(m) * n, batch,
            CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
        CUDA_CHECK(cudaGetLastError());
    };
    for (int64_t i3 = 0; i3 < b->ne[3]; ++i3) {
        const char * a3 = ap + (i3 / (b->ne[3] / a->ne[3])) * al.nb[3];
        const char * b3 = bp + i3 * bl.nb[3];
        float * output = reinterpret_cast<float *>(static_cast<char *>(dst->data) + i3 * dst->nb[3]);
        if (a->ne[2] == b->ne[2] || a->ne[2] == 1) {
            GGML_ASSERT(b->ne[2] <= INT_MAX);
            gemm(a3, b3, output, a->ne[2] == 1 ? 0 : al.nb[2], bl.nb[2], int(b->ne[2]));
        } else {
            const int64_t repeat = b->ne[2] / a->ne[2];
            GGML_ASSERT(repeat <= INT_MAX);
            for (int64_t group = 0; group < a->ne[2]; ++group)
                gemm(a3 + group * al.nb[2], b3 + group * repeat * bl.nb[2],
                    output + group * repeat * m * n, 0, bl.nb[2], int(repeat));
        }
    }
}
