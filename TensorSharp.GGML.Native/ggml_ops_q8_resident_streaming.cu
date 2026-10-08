// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Dispatch/reduction follows the unchanged ggml ffa4e8b CUDA implementation.
// Its public MMQ device templates and Q8 vec-dot primitive are reused directly;
// all allocations and recoverable launch checks remain TensorSharp-owned.
#include "ggml_ops_q8_resident_streaming.h"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/vecdotq.cuh"
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>

namespace {
void checked(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(std::string("Resident Q8 streaming: ") + cudaGetErrorString(error));
}
std::size_t aligned(std::uint64_t bytes) {
    if (bytes > std::uint64_t(INT64_MAX) - 255) throw std::overflow_error("Resident Q8 workspace overflow.");
    return std::size_t((bytes + 255) & ~std::uint64_t(255));
}
void validate_device(int device) {
    const auto& info = ggml_cuda_info();
    if (device < 0 || device >= info.device_count || info.devices[device].cc < GGML_CUDA_CC_AMPERE ||
        info.devices[device].cc >= GGML_CUDA_CC_BLACKWELL || info.devices[device].warp_size != 32 ||
        !ampere_mma_available(info.devices[device].cc))
        throw std::invalid_argument("Resident CUDA streaming currently supports NVIDIA Ampere, Ada and Hopper.");
}
int select_J(int device, int logical_columns, bool fallback) {
    const auto& info = ggml_cuda_info().devices[device];
    int best = 0, tiles_best = INT_MAX;
    for (int J = 8; J <= 128 && tiles_best > 1; J += 8) {
        const auto config = ggml_cuda_mmq_get_config(GGML_TYPE_Q8_0, J, fallback, info.cc);
        if (config.type == GGML_TYPE_COUNT || config.I != 128 || !config.stream_k ||
            mmq_get_nbytes_shared(config, info.cc) > info.smpbo) continue;
        const int tiles = (logical_columns + config.J - 1) / config.J;
        if (tiles < tiles_best) { best = J; tiles_best = tiles; }
    }
    if (!best) throw std::invalid_argument("Resident Q8 streaming has no supported MMQ configuration.");
    return best;
}
std::size_t quantized_bytes(int inner, int columns) {
    const auto padded = (std::uint64_t(inner) + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING * MATRIX_ROW_PADDING;
    // MMQ reserves a complete maximum-J block after the quantized matrix for
    // vectorized tile reads. MMVQ uses a prefix of the same charged workspace.
    return aligned(padded / QK8_1 * columns * sizeof(block_q8_1) + 128 * sizeof(block_q8_1_mmq));
}

// Q8_1's vector format uses an FP16 scale, unlike MMQ's Q8_0 D4 layout.
// Preserve upstream division/roundf and warp reductions, with the same fast
// math compilation policy. This kernel has no aborting host launch wrapper.
__global__ void quantize_vector(const float* input, block_q8_1* output, int inner, int padded) {
    const int k = int(blockIdx.x) * 256 + threadIdx.x;
    if (k >= padded) return;
    const float value = k < inner ? input[std::size_t(blockIdx.y) * inner + k] : 0.0f;
    const float maximum = warp_reduce_max<32>(fabsf(value));
    const float sum = warp_reduce_sum<32>(value);
    const float d = maximum / 127.0f;
    auto& block = output[std::size_t(blockIdx.y) * (padded / 32) + k / 32];
    block.qs[k % 32] = maximum == 0 ? 0 : int8_t(roundf(value / d));
    if (k % 32 == 0) block.ds = make_half2(d, sum);
}

template<int N, int Warps, int Rows>
__global__ void vector_projection(const void* weights, const block_q8_1* input, float* output,
        int inner, int padded, int rows) {
    const int tid = int(threadIdx.y) * 32 + threadIdx.x;
    const int first = int(blockIdx.x) * Rows;
    const int blocks = inner / 32;
    float sums[N][Rows] = {};
    for (int block = tid / 4; block < blocks; block += Warps * 8) {
#pragma unroll
        for (int n = 0; n < N; ++n)
#pragma unroll
            for (int row = 0; row < Rows; ++row)
                sums[n][row] += vec_dot_q8_0_q8_1(weights, input + n * (padded / 32) + block,
                    (first + row) * blocks + block, 2 * (tid % 4));
    }
    __shared__ float partial[Warps - 1][N][Rows][32];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int n = 0; n < N; ++n)
#pragma unroll
            for (int row = 0; row < Rows; ++row) partial[threadIdx.y - 1][n][row][threadIdx.x] = sums[n][row];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int n = 0; n < N; ++n)
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
#pragma unroll
            for (int warp = 0; warp < Warps - 1; ++warp) sums[n][row] += partial[warp][n][row][threadIdx.x];
            const float value = warp_reduce_sum<32>(sums[n][row]);
            if (threadIdx.x == row && first + row < rows) output[std::size_t(n) * rows + first + row] = value;
        }
}
template<int N> void launch_vector(const void* weights, const block_q8_1* input, float* output,
        int inner, int padded, int rows, int logical_columns, cudaStream_t stream) {
    if (logical_columns <= 4) {
        if (logical_columns == 1 && inner < 1024)
            vector_projection<N, 4, 4><<<unsigned((rows + 3) / 4), dim3(32, 4), 0, stream>>>(weights, input, output, inner, padded, rows);
        else if (logical_columns == 1)
            vector_projection<N, 4, 1><<<rows, dim3(32, 4), 0, stream>>>(weights, input, output, inner, padded, rows);
        else
            vector_projection<N, 4, 2><<<unsigned((rows + 1) / 2), dim3(32, 4), 0, stream>>>(weights, input, output, inner, padded, rows);
    } else {
        vector_projection<N, 2, 2><<<unsigned((rows + 1) / 2), dim3(32, 2), 0, stream>>>(weights, input, output, inner, padded, rows);
    }
}

template<int J, bool Fallback>
void launch_matrix(int device, const void* weights, const int* input, float* output, float* fixup,
        int inner, int padded, int rows, int columns, cudaStream_t stream) {
    const auto& info = ggml_cuda_info().devices[device];
    const auto config = ggml_cuda_mmq_get_config(GGML_TYPE_Q8_0, J, Fallback, info.cc);
    const int shared = int(mmq_get_nbytes_shared(config, info.cc));
    checked(cudaFuncSetAttribute(mul_mat_q<GGML_TYPE_Q8_0, J, Fallback>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    const int nx = (columns + J - 1) / J, ny = (rows + config.I - 1) / config.I;
    const int tiles = nx * ny;
    const int waves = (tiles + info.nsm - 1) / info.nsm;
    const int blocks = 100LL * tiles / (info.nsm * waves) >= 90 ? tiles : info.nsm;
    const bool needs_fixup = tiles % blocks != 0;
    const dim3 threads(32, config.nthreads / 32);
    const auto one = init_fastdiv_values(1), k = init_fastdiv_values(inner / 32), x = init_fastdiv_values(nx);
    const int input_stride = columns * (padded / 32) * int(sizeof(block_q8_1) / sizeof(int));
    mul_mat_q<GGML_TYPE_Q8_0, J, Fallback><<<blocks, threads, shared, stream>>>(
        static_cast<const char*>(weights), input, nullptr, nullptr, output, needs_fixup ? fixup : nullptr, nullptr,
        k, rows, columns, inner / 32, columns, rows, one, one, rows * (inner / 32), input_stride, rows * columns,
        one, one, rows * (inner / 32), input_stride, rows * columns, x);
    checked(cudaGetLastError());
    if (needs_fixup) {
        mul_mat_q_stream_k_fixup<GGML_TYPE_Q8_0, J, Fallback><<<dim3(blocks, config.I / 32),
            dim3(32, config.nthreads / 64), 0, stream>>>(nullptr, nullptr, output, fixup, k, rows, columns, rows,
                one, rows * columns, one, rows * columns, x);
        checked(cudaGetLastError());
    }
}
}

int tsg_q8_resident_output_rows(int rows) {
    if (rows <= 0 || rows > INT_MAX - 127) throw std::invalid_argument("Resident Q8 output rows exceed padded integer range.");
    return (rows + 127) / 128 * 128;
}
tsg_q8_resident_layout tsg_q8_resident_sizes(int device, int inner, int rows, int columns,
        int logical_columns, std::int64_t logical_rows) {
    validate_device(device);
    if (inner <= 0 || inner > INT_MAX - 511 || inner % 32 || columns <= 0 || columns > 65535 ||
        logical_columns < columns || logical_columns > 524280 || logical_rows < rows || logical_rows > INT_MAX)
        throw std::invalid_argument("Resident Q8 logical shape or workspace dimensions are unsupported.");
    const int padded_rows = tsg_q8_resident_output_rows(rows);
    const int padded = (inner + 511) / 512 * 512;
    if (logical_columns > 8) {
        checked(cudaSetDevice(device));
        if (!ggml_cuda_should_use_mmq(GGML_TYPE_Q8_0, ggml_cuda_info().devices[device].cc, logical_columns, 0))
            throw std::invalid_argument("Resident Q8 compatibility requires the upstream MMQ policy for matrix batches.");
    }
    const int J = logical_columns > 8 ? select_J(device, logical_columns, logical_rows % 128 != 0) : 8;
    const std::uint64_t tiles = std::uint64_t((columns + J - 1) / J) * (padded_rows / 128);
    if (tiles * (inner / 32) >= (1ULL << 30) || std::uint64_t(padded_rows) * columns > INT_MAX ||
        std::uint64_t(padded_rows) * (inner / 32) > INT_MAX || std::uint64_t(columns) * (padded / 32) * 9 > INT_MAX)
        throw std::invalid_argument("Resident Q8 dimensions exceed upstream kernel indexing limits.");
    const auto fixup = logical_columns > 8 ? aligned(std::uint64_t(ggml_cuda_info().devices[device].nsm) * 128 * J * sizeof(float)) : 0;
    return {aligned(std::uint64_t(inner / 32) * 34 * padded_rows + std::uint64_t((padded - inner) / 32) * 34),
        aligned(std::uint64_t(padded_rows) * columns * sizeof(float)), quantized_bytes(inner, columns) + fixup};
}

void tsg_q8_resident_launch(int device, const void* weights, const float* input, float* output,
        void* scratch, std::size_t scratch_bytes, int inner, int rows, int columns,
        int logical_columns, std::int64_t logical_rows, void* stream_pointer) {
    const auto expected = tsg_q8_resident_sizes(device, inner, rows, columns, logical_columns, logical_rows);
    if (!scratch || scratch_bytes < expected.scratch_bytes) throw std::invalid_argument("Resident Q8 scratch is not fully reserved.");
    const int padded = (inner + 511) / 512 * 512, padded_rows = tsg_q8_resident_output_rows(rows);
    const auto stream = static_cast<cudaStream_t>(stream_pointer);
    if (logical_columns <= 8) {
        quantize_vector<<<dim3(unsigned(padded / 256), unsigned(columns)), 256, 0, stream>>>(input,
            static_cast<block_q8_1*>(scratch), inner, padded);
        checked(cudaGetLastError());
#define VECTOR_CASE(N) case N: launch_vector<N>(weights, static_cast<const block_q8_1*>(scratch), output, inner, padded, padded_rows, logical_columns, stream); break
        switch (columns) { VECTOR_CASE(1); VECTOR_CASE(2); VECTOR_CASE(3); VECTOR_CASE(4); VECTOR_CASE(5); VECTOR_CASE(6); VECTOR_CASE(7); VECTOR_CASE(8); }
#undef VECTOR_CASE
        checked(cudaGetLastError());
    } else {
        // This upstream helper only submits quantization; it allocates nothing
        // and has no CUDA_CHECK host wrapper. Preconditions were checked above.
        quantize_mmq_q8_1_cuda(input, nullptr, scratch, GGML_TYPE_Q8_0, inner, inner,
            std::int64_t(inner) * columns, std::int64_t(inner) * columns, padded, columns, 1, 1, stream);
        checked(cudaGetLastError());
        const bool fallback = logical_rows % 128 != 0;
        const int J = select_J(device, logical_columns, fallback);
        auto* fixup = reinterpret_cast<float*>(static_cast<char*>(scratch) + quantized_bytes(inner, columns));
#define MATRIX_CASE(J) case J: if (fallback) launch_matrix<J, true>(device, weights, static_cast<const int*>(scratch), output, fixup, inner, padded, padded_rows, columns, stream); else launch_matrix<J, false>(device, weights, static_cast<const int*>(scratch), output, fixup, inner, padded, padded_rows, columns, stream); break
        switch (J) { MATRIX_CASE(8); MATRIX_CASE(16); MATRIX_CASE(24); MATRIX_CASE(32); MATRIX_CASE(40); MATRIX_CASE(48); MATRIX_CASE(56); MATRIX_CASE(64); MATRIX_CASE(72); MATRIX_CASE(80); MATRIX_CASE(88); MATRIX_CASE(96); MATRIX_CASE(104); MATRIX_CASE(112); MATRIX_CASE(120); MATRIX_CASE(128); }
#undef MATRIX_CASE
    }
}
