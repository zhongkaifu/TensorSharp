// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Opt-in numerical/performance investigation only. Never production dispatch.
#include "ggml_ops_q8_precision.h"
#include "precision_test_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <string>

namespace {
void checked(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }

// One warp owns one output row. Unlike the production K-ordered kernel,
// independent lane sums change the reduction order. F32 and F64 variants are
// both compared with the mathematical dot, not declared correct by old parity.
template<class Accumulator>
__global__ void parallel_k(const char * weights, const float * input, float * output,
        int k, int m, size_t weight_stride) {
    const int lane = int(threadIdx.x) & 31;
    const int row = int(blockIdx.x) * 4 + int(threadIdx.x) / 32;
    if (row >= m) return; // whole warp takes the same branch
    Accumulator sum = 0;
    for (int block = 0; block < k / 32; ++block) {
        const char * source = weights + size_t(row) * weight_stride + size_t(block) * 34;
        const float scale = __half2float(*reinterpret_cast<const half *>(source));
        const float weight = scale * float(*reinterpret_cast<const int8_t *>(source + 2 + lane));
        const float activation = input[block * 32 + lane];
        if constexpr (sizeof(Accumulator) == sizeof(double))
            sum = fma(double(weight), double(activation), sum);
        else
            sum = fmaf(weight, activation, sum);
    }
    for (int shift = 16; shift > 0; shift /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, shift);
    if (lane == 0) output[row] = float(sum);
}

float scale(int row, int block) {
    return std::ldexp(1.0f + float((row * 71 + block * 19) % 1024) / 1024.0f, -7 + (row + block) % 4);
}
int quant(int row, int k) { return (k * 37 + row * 19) % 256 - 128; }

struct DeviceAllocation {
    void * pointer = nullptr;
    explicit DeviceAllocation(size_t bytes) { checked(cudaMalloc(&pointer, bytes)); }
    ~DeviceAllocation() { if (pointer) cudaFree(pointer); }
    DeviceAllocation(const DeviceAllocation &) = delete;
    DeviceAllocation & operator=(const DeviceAllocation &) = delete;
};

struct Metrics { double relative, maximum, rounding_bound_ratio; size_t differing_serial; };

void measure(int k, int m, bool cancellation) {
    const size_t stride = size_t(k / 32) * 34;
    std::vector<unsigned char> weights(size_t(m) * stride);
    std::vector<float> input(k);
    for (int row = 0; row < m; ++row) for (int block = 0; block < k / 32; ++block) {
        const uint16_t bits = cancellation ? 0x3c00 :
            uint16_t(((8 + (row + block) % 4) << 10) | ((row * 71 + block * 19) % 1024));
        auto * destination = weights.data() + size_t(row) * stride + size_t(block) * 34;
        std::memcpy(destination, &bits, 2);
        for (int j = 0; j < 32; ++j) {
            const int8_t value = int8_t(cancellation ? 1 : quant(row, block * 32 + j));
            std::memcpy(destination + 2 + j, &value, 1);
        }
    }
    for (int i = 0; i < k; ++i) {
        if (cancellation) input[i] = i % 4 == 0 ? 100000000.0f : i % 4 == 2 ? -100000000.0f : 1.0f;
        else {
            const uint32_t hash = uint32_t(i + 1) * 2654435761u ^ 2246822519u;
            input[i] = float(int(hash % 65537) - 32768) / 32768.0f + float((hash >> 17) % 13) * 0x1p-22f;
        }
    }
    std::set<int> samples{0, m - 1, m / 2};
    for (int i = 0; i < 61; ++i) samples.insert(int(int64_t(i) * (m - 1) / 60));
    std::vector<double> expected, absolute_products;
    for (int row : samples) {
        double dot = 0, absolute = 0;
        for (int j = 0; j < k; ++j) {
            const double weight = cancellation ? 1 : double(scale(row, j / 32)) * quant(row, j);
            const double product = weight * input[j];
            dot += product; absolute += std::abs(product);
        }
        expected.push_back(dot); absolute_products.push_back(absolute);
    }
    DeviceAllocation device_weights(weights.size()), device_input(input.size() * sizeof(float));
    const size_t output_bytes = (size_t(m) + 2) * sizeof(float);
    DeviceAllocation serial_output(output_bytes), parallel_output(output_bytes), double_output(output_bytes);
    void * destinations[] = {serial_output.pointer, parallel_output.pointer, double_output.pointer};
    const char * names[] = {"qualified_serial_f32", "parallel32_f32", "parallel32_f64"};
    checked(cudaMemcpy(device_weights.pointer, weights.data(), weights.size(), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(device_input.pointer, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<float> canaries(size_t(m) + 2, -12345.625f);
    for (void * destination : destinations)
        checked(cudaMemcpy(destination, canaries.data(), output_bytes, cudaMemcpyHostToDevice));
    auto launch = [&](int route) {
        auto * result = static_cast<float *>(destinations[route]) + 1;
        if (route == 0) checked(static_cast<cudaError_t>(tsg_matmul_q8_cuda_launch(device_weights.pointer,
            device_input.pointer, result, k, m, 1, stride, sizeof(float), input.size() * sizeof(float), nullptr)));
        else {
            const unsigned blocks = unsigned((int64_t(m) + 3) / 4);
            if (route == 1) parallel_k<float><<<blocks, 128>>>(static_cast<const char *>(device_weights.pointer),
                static_cast<const float *>(device_input.pointer), result, k, m, stride);
            else parallel_k<double><<<blocks, 128>>>(static_cast<const char *>(device_weights.pointer),
                static_cast<const float *>(device_input.pointer), result, k, m, stride);
            checked(cudaGetLastError());
        }
    };
    for (int warmup = 0; warmup < 3; ++warmup) for (int route = 0; route < 3; ++route) launch(route);
    checked(cudaDeviceSynchronize());
    std::array<Metrics, 3> metrics;
    std::vector<float> reference_output, actual(size_t(m) + 2);
    for (int route = 0; route < 3; ++route) {
        checked(cudaMemcpy(actual.data(), destinations[route], output_bytes, cudaMemcpyDeviceToHost));
        require(actual.front() == canaries.front() && actual.back() == canaries.back(), "Parallel Q8 output canary overwritten");
        for (int row = 0; row < m; ++row) require(std::isfinite(actual[size_t(row) + 1]), "Nonfinite parallel Q8 output");
        if (route == 0) reference_output = actual;
        double error_squared = 0, reference_squared = 0, maximum = 0, maximum_bound_ratio = 0;
        size_t index = 0, differing = 0;
        for (int row = 0; row < m; ++row)
            if (std::memcmp(&actual[size_t(row) + 1], &reference_output[size_t(row) + 1], sizeof(float)) != 0) ++differing;
        for (int row : samples) {
            const double error = std::abs(double(actual[size_t(row) + 1]) - expected[index]);
            const double terms = route == 0 ? k : k / 32 + 5;
            const double nu = terms * (route == 2 ? 0x1p-53 : 0x1p-24);
            // Tree reduction has max lane-depth K/32 plus five additions.
            // Double accumulation still has a final F32 output rounding.
            const double oracle_nu = k * 0x1p-53;
            const double bound = (nu / (1 - nu) + oracle_nu / (1 - oracle_nu)) * absolute_products[index] +
                (route == 2 ? 0x1p-24 * std::abs(expected[index]) : 0);
            require(error <= bound, "Parallel Q8 sampled FP64 forward-error bound failed");
            maximum_bound_ratio = std::max(maximum_bound_ratio, error / std::max(bound, 1e-300));
            error_squared += error * error; reference_squared += expected[index] * expected[index];
            maximum = std::max(maximum, error); ++index;
        }
        metrics[route] = {std::sqrt(error_squared / std::max(reference_squared, 1e-300)), maximum, maximum_bound_ratio, differing};
    }
    cudaEvent_t start, end; checked(cudaEventCreate(&start)); checked(cudaEventCreate(&end));
    for (int pass = 0; pass < 6; ++pass) {
        const int route = pass < 3 ? pass : 5 - pass; // A/B/C/C/B/A
        checked(cudaEventRecord(start));
        for (int i = 0; i < 3; ++i) launch(route);
        checked(cudaEventRecord(end)); checked(cudaEventSynchronize(end));
        float elapsed; checked(cudaEventElapsedTime(&elapsed, start, end));
        const int repetitions = std::clamp(int(std::ceil(100.0 / std::max(double(elapsed) / 3, 0.001))), 3, 10000);
        checked(cudaEventRecord(start));
        for (int i = 0; i < repetitions; ++i) launch(route);
        checked(cudaEventRecord(end)); checked(cudaEventSynchronize(end));
        checked(cudaEventElapsedTime(&elapsed, start, end));
        const Metrics & result = metrics[route];
        std::printf("{\"data\":\"%s\",\"production_candidate\":false,\"k\":%d,\"m\":%d,\"n\":1,\"route\":\"%s\",\"pass\":%d,\"repetitions\":%d,\"stream_us\":%.6f,\"payload_bytes\":%zu,\"different_from_serial_rows\":%zu,\"oracle_samples\":%zu,\"oracle_max_error\":%.9g,\"oracle_relative_l2\":%.9g,\"oracle_max_rounding_bound_ratio\":%.9g}\n",
            cancellation ? "synthetic_cancellation" : "synthetic", k, m, names[route], pass, repetitions,
            double(elapsed) * 1000 / repetitions, weights.size() + input.size() * sizeof(float) + 3 * output_bytes,
            result.differing_serial, samples.size(), result.maximum, result.relative, result.rounding_bound_ratio);
    }
    checked(cudaEventDestroy(start)); checked(cudaEventDestroy(end));
}
}

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char * experimental = std::getenv("TS_GGML_Q8_PARALLEL_VECTOR");
    require(!experimental || std::strcmp(experimental, "1") != 0,
        "Unset TS_GGML_Q8_PARALLEL_VECTOR so the qualified control keeps K-ordered arithmetic");
    if (argc < 2 || std::string(argv[1]) != "--benchmark" || (argc != 2 && argc != 4)) {
        std::fprintf(stderr, "Opt-in: --benchmark [K M]; synthetic numerical investigation, no production dispatch.\n"); return 2;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    checked(cudaSetDevice(0));
    if (argc == 4) {
        const int k = std::atoi(argv[2]), m = std::atoi(argv[3]);
        require(k > 0 && k <= 8192 && k % 32 == 0 && m > 0 && m <= 262144, "Invalid bounded benchmark shape");
        measure(k, m, false);
    } else {
        // Exact Qwen3.5-0.8B projection geometries, plus larger output heads.
        const int shapes[][2] = {{1024, 5120}, {1024, 8224}, {2048, 1024},
            {1024, 7168}, {3584, 1024}, {1024, 248320}, {2560, 248320}, {4096, 248320}};
        for (const auto & shape : shapes)
            measure(shape[0], shape[1], false);
        // Large cancellation deliberately challenges the existing serial sum;
        // this is an oracle test, not a requirement to reproduce its rounding.
        measure(4096, 33, true);
    }
    return 0;
}
