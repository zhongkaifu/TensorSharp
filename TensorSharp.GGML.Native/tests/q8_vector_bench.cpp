// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Explicit synthetic-data microbenchmark at model projection shapes, not a CTest.
#include "ggml_ops_q8_precision.h"
#include "precision_test_utils.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <set>
#include <string>

namespace {
void checked(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }
float scale(int row, int block) {
    return std::ldexp(1.0f + float((row * 71 + block * 19) % 1024) / 1024.0f, -7 + (row + block) % 4);
}
int quant(int row, int k) { return (k * 37 + row * 19) % 256 - 128; }

void measure(int k, int m) {
    const size_t stride = size_t(k / 32) * 34, weight_bytes = size_t(m) * stride;
    std::vector<unsigned char> weights(weight_bytes);
    std::vector<float> input(k), old_output(size_t(m) + 2, -12345.625f), new_output(old_output);
    for (int row = 0; row < m; ++row) for (int block = 0; block < k / 32; ++block) {
        const uint16_t bits = uint16_t(((8 + (row + block) % 4) << 10) | ((row * 71 + block * 19) % 1024));
        auto * destination = weights.data() + size_t(row) * stride + size_t(block) * 34;
        std::memcpy(destination, &bits, 2);
        for (int j = 0; j < 32; ++j) {
            const int8_t value = int8_t(quant(row, block * 32 + j));
            std::memcpy(destination + 2 + j, &value, 1);
        }
    }
    for (int i = 0; i < k; ++i) {
        const uint32_t hash = uint32_t(i + 1) * 2654435761u ^ 2246822519u;
        input[i] = float(int(hash % 65537) - 32768) / 32768.0f + float((hash >> 17) % 13) * 0x1p-22f;
    }
    void * device_weights = nullptr;
    float * device_input = nullptr, * old_device = nullptr, * new_device = nullptr;
    const size_t output_bytes = old_output.size() * sizeof(float);
    checked(cudaMalloc(&device_weights, weight_bytes));
    checked(cudaMalloc(reinterpret_cast<void **>(&device_input), input.size() * sizeof(float)));
    checked(cudaMalloc(reinterpret_cast<void **>(&old_device), output_bytes));
    checked(cudaMalloc(reinterpret_cast<void **>(&new_device), output_bytes));
    checked(cudaMemcpy(device_weights, weights.data(), weight_bytes, cudaMemcpyHostToDevice));
    checked(cudaMemcpy(device_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(old_device, old_output.data(), output_bytes, cudaMemcpyHostToDevice));
    checked(cudaMemcpy(new_device, new_output.data(), output_bytes, cudaMemcpyHostToDevice));
    auto launch = [&](bool previous) {
        auto function = previous ? tsg_matmul_q8_cuda_launch_reference : tsg_matmul_q8_cuda_launch;
        checked(static_cast<cudaError_t>(function(device_weights, device_input,
            (previous ? old_device : new_device) + 1, k, m, 1, stride, sizeof(float),
            size_t(k) * sizeof(float), nullptr)));
    };
    for (int warmup = 0; warmup < 3; ++warmup) { launch(true); launch(false); }
    checked(cudaDeviceSynchronize());
    checked(cudaMemcpy(old_output.data(), old_device, output_bytes, cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(new_output.data(), new_device, output_bytes, cudaMemcpyDeviceToHost));
    require(std::memcmp(old_output.data(), new_output.data(), output_bytes) == 0, "Old/new Q8 decode output differs bitwise");
    require(new_output.front() == -12345.625f && new_output.back() == -12345.625f, "Q8 decode output canary overwritten");
    for (int row = 0; row < m; ++row) require(std::isfinite(new_output[size_t(row) + 1]), "Nonfinite Q8 decode output");
    std::set<int> samples{0, m - 1, m / 2};
    for (int i = 0; i < 61; ++i) samples.insert(int(int64_t(i) * (m - 1) / 60));
    double error_squared = 0, reference_squared = 0, maximum = 0, maximum_rounding_bound_ratio = 0;
    for (int row : samples) {
        double expected = 0, absolute_products = 0;
        for (int j = 0; j < k; ++j) {
            const double product = double(scale(row, j / 32)) * quant(row, j) * input[j];
            expected += product; absolute_products += std::abs(product);
        }
        const double error = std::abs(double(new_output[size_t(row) + 1]) - expected);
        // A timing fixture may cancel almost completely. Use the standard
        // sequential-FMA forward-error bound, not an arbitrary near-zero
        // relative tolerance. Every decoded half*int8 weight is exact in F32.
        // The standalone correctness suite retains its stricter fixed gates.
        const double ku = k * 0x1p-24;
        const double bound = (ku / (1 - ku)) * absolute_products;
        require(error <= bound, "Q8 decode sampled FP64 rounding bound failed");
        maximum_rounding_bound_ratio = std::max(maximum_rounding_bound_ratio, error / std::max(bound, 1e-300));
        error_squared += error * error; reference_squared += expected * expected;
        maximum = std::max(maximum, error);
    }
    const double relative = std::sqrt(error_squared / std::max(reference_squared, 1e-300));
    cudaEvent_t start, end; checked(cudaEventCreate(&start)); checked(cudaEventCreate(&end));
    for (int pass = 0; pass < 4; ++pass) {
        const bool previous = pass == 0 || pass == 3; // old/new/new/old, same resident buffers
        checked(cudaEventRecord(start));
        for (int i = 0; i < 3; ++i) launch(previous);
        checked(cudaEventRecord(end)); checked(cudaEventSynchronize(end));
        float elapsed; checked(cudaEventElapsedTime(&elapsed, start, end));
        int repetitions = std::clamp(int(std::ceil(100.0 / std::max(double(elapsed) / 3, 0.001))), 3, 10000);
        auto wall = std::chrono::steady_clock::now();
        checked(cudaEventRecord(start));
        for (int i = 0; i < repetitions; ++i) launch(previous);
        checked(cudaEventRecord(end)); checked(cudaEventSynchronize(end));
        const double wall_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - wall).count() / repetitions;
        checked(cudaEventElapsedTime(&elapsed, start, end));
        std::printf("{\"data\":\"synthetic\",\"k\":%d,\"m\":%d,\"n\":1,\"route\":\"%s\",\"pass\":%d,\"repetitions\":%d,\"stream_us\":%.6f,\"wall_us\":%.6f,\"payload_bytes\":%zu,\"bitwise_equal\":true,\"oracle_samples\":%zu,\"oracle_max_error\":%.9g,\"oracle_relative_l2\":%.9g,\"oracle_max_fma_rounding_bound_ratio\":%.9g}\n",
            k, m, previous ? "previous_matrix8" : "vector1", pass, repetitions, double(elapsed) * 1000 / repetitions,
            wall_us, weight_bytes + input.size() * sizeof(float) + 2 * output_bytes, samples.size(), maximum, relative, maximum_rounding_bound_ratio);
    }
    checked(cudaEventDestroy(start)); checked(cudaEventDestroy(end));
    checked(cudaFree(new_device)); checked(cudaFree(old_device)); checked(cudaFree(device_input)); checked(cudaFree(device_weights));
}
}

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char * experimental = std::getenv("TS_GGML_Q8_PARALLEL_VECTOR");
    require(!experimental || std::strcmp(experimental, "1") != 0,
        "Unset TS_GGML_Q8_PARALLEL_VECTOR: this benchmark compares the qualified K-ordered kernel");
    if (argc < 2 || std::string(argv[1]) != "--benchmark" || (argc != 2 && argc != 4)) {
        std::fprintf(stderr, "Opt-in: --benchmark [K M]; synthetic inputs at model matrix shapes.\n"); return 2;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    checked(cudaSetDevice(0));
    if (argc == 4) {
        int k = std::atoi(argv[2]), m = std::atoi(argv[3]);
        require(k > 0 && k <= 8192 && k % 32 == 0 && m > 0 && m <= 262144, "Invalid bounded benchmark shape");
        measure(k, m);
    } else {
        for (auto shape : {std::array<int, 2>{1024, 5120}, {1024, 8224}, {1024, 6144}, {3072, 1024},
                           {1024, 248320}, {2560, 248320}, {4096, 248320}}) measure(shape[0], shape[1]);
    }
    return 0;
}
