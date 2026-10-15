// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Small diagnostic against the unchanged ggml CUDA quantizer linked into GgmlOps.
// This reports sensitivity; it is not a model or correctness qualification test.
#include "ggml.h"
#include "ggml-cuda.h"
#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

// Dependency-internal host launcher: signature pinned to ggml ffa4e8b8.
void quantize_row_q8_1_cuda(const float*, const std::int32_t*, void*, ggml_type,
    std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    std::int64_t, std::int64_t, std::int64_t, std::int64_t, cudaStream_t);

static void checked(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

int main() {
    try {
        auto* backend = ggml_backend_cuda_init(0);
        if (!backend) return 2;
        float* input = nullptr;
        void* packed = nullptr;
        checked(cudaMalloc(&input, 256 * sizeof(float)));
        checked(cudaMalloc(&packed, 8 * 36)); // Q8_1: half2 scales + 32 signed values.
        for (float maximum : {3.1187703609466553f, 3.1187708377838135f}) {
            std::array<float, 256> values{};
            values[18] = -0.6507671475410461f;
            values[26] = maximum;
            checked(cudaMemcpy(input, values.data(), sizeof(values), cudaMemcpyHostToDevice));
            quantize_row_q8_1_cuda(input, nullptr, packed, GGML_TYPE_Q8_0,
                32, 32, 32, 32, 256, 1, 1, 1, nullptr);
            std::array<std::int8_t, 8 * 36> result{};
            checked(cudaMemcpy(result.data(), packed, result.size(), cudaMemcpyDeviceToHost));
            std::printf("maximum=%.9g unchanged_activation=%.9g quantized_activation=%d\n",
                maximum, values[18], int(result[4 + 18]));
        }
        checked(cudaFree(packed));
        checked(cudaFree(input));
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
