// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_matmul_precision.h"
#include "ggml_ops_dsv4_fused.h"
#include "ggml_ops_precision_policy.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-cuda/common.cuh"
#include <algorithm>
#include <climits>

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
    if (needed > state->capacity) {
        // Custom ops execute between the CUDA backend's captured graph views.
        // This storage persists across nodes/steps; growth is outside capture
        // and waits for earlier users before replacing their allocation.
        cudaStreamCaptureStatus capture;
        CUDA_CHECK(cudaStreamIsCapturing(state->stream, &capture));
        GGML_ASSERT(capture == cudaStreamCaptureStatusNone);
        CUDA_CHECK(cudaStreamSynchronize(state->stream));
        if (state->scratch) CUDA_CHECK(cudaFree(state->scratch));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state->scratch), needed));
        state->capacity = needed;
    }
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
