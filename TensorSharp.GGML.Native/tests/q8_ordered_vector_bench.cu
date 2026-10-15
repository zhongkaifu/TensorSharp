// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Synthetic-data investigation. These kernels are not production dispatch.
#include "ggml_ops_q8_precision.h"
#include "precision_test_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <set>
#include <string>

namespace {
void checked(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }

// Each lane owns one complete dot. Independent rows hide its FMA dependency;
// L1 serves adjacent bytes in each row across successive K instructions.
__global__ void row_thread(const char * w, const char * x, float * y,
        int k, int m, size_t ws, size_t xs) {
    const int row = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (row >= m) return;
    float sum = 0;
    for (int b = 0; b < k / 32; ++b) {
        const char * block = w + size_t(row) * ws + size_t(b) * 34;
        float scale = __half2float(*reinterpret_cast<const half *>(block));
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            float weight = scale * float(*reinterpret_cast<const int8_t *>(block + 2 + j));
            float activation = *reinterpret_cast<const float *>(x + size_t(b * 32 + j) * xs);
            sum = fmaf(weight, activation, sum);
        }
    }
    y[row] = sum;
}

template<int Rows>
__global__ void row_packed(const char * w, const char * x, float * y,
        int k, int m, size_t ws, size_t xs) {
    const int first = int(blockIdx.x) * int(blockDim.x) * Rows + int(threadIdx.x);
    if (first >= m) return;
    float sum[Rows] = {};
    for (int b = 0; b < k / 32; ++b) {
        float scale[Rows];
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            int row = first + r * int(blockDim.x);
            scale[r] = row < m ? __half2float(*reinterpret_cast<const half *>(w + size_t(row) * ws + size_t(b) * 34)) : 0;
        }
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            float a = *reinterpret_cast<const float *>(x + size_t(b * 32 + j * 2) * xs);
            float c = *reinterpret_cast<const float *>(x + size_t(b * 32 + j * 2 + 1) * xs);
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                int row = first + r * int(blockDim.x);
                if (row < m) {
                    // Q8_0 has only two-byte alignment, including its 34-byte
                    // block stride. Wider typed loads would not be safe.
                    uint16_t packed = *reinterpret_cast<const uint16_t *>(w + size_t(row) * ws + size_t(b) * 34 + 2 + j * 2);
                    sum[r] = fmaf(scale[r] * float(int8_t(packed & 255)), a, sum[r]);
                    sum[r] = fmaf(scale[r] * float(int8_t(packed >> 8)), c, sum[r]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r)
        if (first + r * int(blockDim.x) < m) y[first + r * int(blockDim.x)] = sum[r];
}

// Warp-private transposition, followed by one or two ordered row dots per
// lane. Unlike parallel-K, no lane contributes a partial reduction. Warp
// synchronization cannot be skipped by tail rows that still supply loads.
template<int Rows, int Warps>
__global__ void warp_rows(const char * w, const char * x, float * y,
        int k, int m, size_t ws, size_t xs) {
    __shared__ float weights[Warps][32][Rows + 1];
    __shared__ float inputs[Warps][32];
    const int lane = int(threadIdx.x) & 31, warp = int(threadIdx.x) / 32;
    const int base = (int(blockIdx.x) * Warps + warp) * Rows;
    if (base >= m) return; // Whole warp, never an individual tail lane.
    float sum[Rows / 32] = {};
    for (int b = 0; b < k / 32; ++b) {
        for (int row = 0; row < Rows; ++row) {
            float weight = 0;
            if (base + row < m) {
                const char * block = w + size_t(base + row) * ws + size_t(b) * 34;
                weight = __half2float(*reinterpret_cast<const half *>(block))
                    * float(*reinterpret_cast<const int8_t *>(block + 2 + lane));
            }
            weights[warp][lane][row] = weight;
        }
        inputs[warp][lane] = *reinterpret_cast<const float *>(x + size_t(b * 32 + lane) * xs);
        __syncwarp();
#pragma unroll
        for (int j = 0; j < 32; ++j) {
#pragma unroll
            for (int r = 0; r < Rows / 32; ++r)
                sum[r] = fmaf(weights[warp][j][lane + r * 32], inputs[warp][j], sum[r]);
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < Rows / 32; ++r)
        if (base + lane + r * 32 < m) y[base + lane + r * 32] = sum[r];
}

struct Allocation {
    void * p = nullptr;
    explicit Allocation(size_t n) { checked(cudaMalloc(&p, n)); }
    ~Allocation() { if (p) cudaFree(p); }
    Allocation(const Allocation &) = delete;
    Allocation & operator=(const Allocation &) = delete;
};

constexpr const char * names[] = {"control", "row_thread", "warp32x1", "warp32x4", "warp64x1", "warp64x4",
    "packed1x32", "packed1x128", "packed2x32", "packed4x32"};
constexpr int routes = int(sizeof(names) / sizeof(names[0]));

void measure(int k, int m, bool padded, bool benchmark) {
    const size_t ws = size_t(k / 32) * 34 + (padded ? 6 : 0), xs = padded ? 8 : 4;
    const size_t weight_bytes = size_t(m) * ws, output_bytes = (size_t(m) + 2) * 4;
    require(weight_bytes <= size_t(2) * 1024 * 1024 * 1024, "Weight fixture exceeds 2 GiB bound");
    std::vector<unsigned char> weights(weight_bytes, 0xff), input(size_t(k) * xs, 0xff);
    std::vector<float> logical(k), control(size_t(m) + 2, -12345.625f), actual(control);
    for (int r = 0; r < m; ++r) for (int b = 0; b < k / 32; ++b) {
        uint16_t scale = uint16_t(((8 + (r + b) % 4) << 10) | ((r * 71 + b * 19) % 1024));
        auto * block = weights.data() + size_t(r) * ws + size_t(b) * 34;
        std::memcpy(block, &scale, 2);
        for (int j = 0; j < 32; ++j) block[2 + j] = uint8_t((uint32_t(b * 32 + j) * 37 + uint32_t(r) * 19) % 256 - 128);
    }
    for (int j = 0; j < k; ++j) {
        uint32_t hash = uint32_t(j + 1) * 2654435761u ^ 2246822519u;
        logical[j] = float(int(hash % 65537) - 32768) / 32768.0f + float((hash >> 17) % 13) * 0x1p-22f;
        std::memcpy(input.data() + size_t(j) * xs, &logical[j], 4);
    }
    Allocation dw(weight_bytes), dx(input.size()), dy(output_bytes);
    checked(cudaMemcpy(dw.p, weights.data(), weight_bytes, cudaMemcpyHostToDevice));
    checked(cudaMemcpy(dx.p, input.data(), input.size(), cudaMemcpyHostToDevice));
    auto launch = [&](int route) {
        auto * w = static_cast<const char *>(dw.p), * x = static_cast<const char *>(dx.p);
        float * y = static_cast<float *>(dy.p) + 1;
        if (route == 0) checked(static_cast<cudaError_t>(tsg_matmul_q8_cuda_launch(w, x, y, k, m, 1, ws, xs, input.size(), nullptr)));
        if (route == 1) row_thread<<<(m + 127) / 128, 128>>>(w, x, y, k, m, ws, xs);
        if (route == 2) warp_rows<32, 1><<<(m + 31) / 32, 32>>>(w, x, y, k, m, ws, xs);
        if (route == 3) warp_rows<32, 4><<<(m + 127) / 128, 128>>>(w, x, y, k, m, ws, xs);
        if (route == 4) warp_rows<64, 1><<<(m + 63) / 64, 32>>>(w, x, y, k, m, ws, xs);
        if (route == 5) warp_rows<64, 4><<<(m + 255) / 256, 128>>>(w, x, y, k, m, ws, xs);
        if (route == 6) row_packed<1><<<(m + 31) / 32, 32>>>(w, x, y, k, m, ws, xs);
        if (route == 7) row_packed<1><<<(m + 127) / 128, 128>>>(w, x, y, k, m, ws, xs);
        if (route == 8) row_packed<2><<<(m + 63) / 64, 32>>>(w, x, y, k, m, ws, xs);
        if (route == 9) row_packed<4><<<(m + 127) / 128, 32>>>(w, x, y, k, m, ws, xs);
        checked(cudaGetLastError());
    };
    checked(cudaMemcpy(dy.p, control.data(), output_bytes, cudaMemcpyHostToDevice));
    launch(0); checked(cudaDeviceSynchronize());
    checked(cudaMemcpy(control.data(), dy.p, output_bytes, cudaMemcpyDeviceToHost));
    std::set<int> samples;
    for (int i = 0; i < (benchmark ? 62 : m); ++i)
        samples.insert(benchmark ? int(int64_t(i) * (m - 1) / 61) : i);
    double maximum = 0, squared = 0, norm = 0;
    for (int r : samples) {
        double dot = 0, absolute = 0;
        for (int j = 0; j < k; ++j) {
            double scale = std::ldexp(1.0 + double((r * 71 + j / 32 * 19) % 1024) / 1024.0, -7 + (r + j / 32) % 4);
            double product = scale * (int((uint32_t(j) * 37 + uint32_t(r) * 19) % 256) - 128) * logical[j];
            dot += product; absolute += std::abs(product);
        }
        double error = std::abs(control[size_t(r) + 1] - dot), ku = k * 0x1p-24;
        require(error <= ku / (1 - ku) * absolute, "Independent sequential FMA forward-error bound failed");
        squared += error * error; norm += dot * dot; maximum = std::max(maximum, error);
    }
    for (int route = 0; route < routes; ++route) {
        std::fill(actual.begin(), actual.end(), -12345.625f);
        checked(cudaMemcpy(dy.p, actual.data(), output_bytes, cudaMemcpyHostToDevice));
        launch(route); checked(cudaDeviceSynchronize());
        checked(cudaMemcpy(actual.data(), dy.p, output_bytes, cudaMemcpyDeviceToHost));
        require(std::memcmp(actual.data(), control.data(), output_bytes) == 0, "Ordered candidate differs from original full output or canaries");
        require(actual.front() == -12345.625f && actual.back() == -12345.625f, "Output canary overwritten");
        require(std::all_of(actual.begin(), actual.end(), [](float x) { return std::isfinite(x); }), "Nonfinite output");
    }
    // Source integrity for bounded checks; no extra giant readback in timing fixtures.
    if (!benchmark) {
        std::vector<unsigned char> readback(weight_bytes);
        checked(cudaMemcpy(readback.data(), dw.p, weight_bytes, cudaMemcpyDeviceToHost));
        require(readback == weights, "Weight input modified");
        readback.resize(input.size()); checked(cudaMemcpy(readback.data(), dx.p, input.size(), cudaMemcpyDeviceToHost));
        require(readback == input, "Activation input modified");
    }
    std::printf("{\"check\":true,\"k\":%d,\"m\":%d,\"padded\":%s,\"routes\":%d,\"bitwise_equal\":true,\"oracle_samples\":%zu,\"oracle_max_error\":%.9g,\"oracle_relative_l2\":%.9g}\n",
        k, m, padded ? "true" : "false", routes, samples.size(), maximum, std::sqrt(squared / std::max(norm, 1e-300)));
    if (!benchmark) return;
    cudaEvent_t start, end; checked(cudaEventCreate(&start)); checked(cudaEventCreate(&end));
    for (int pass = 0; pass < routes * 2; ++pass) {
        int route = pass < routes ? pass : routes * 2 - 1 - pass;
        for (int i = 0; i < 3; ++i) launch(route);
        checked(cudaEventRecord(start));
        for (int i = 0; i < 3; ++i) launch(route);
        checked(cudaEventRecord(end)); checked(cudaEventSynchronize(end));
        float elapsed; checked(cudaEventElapsedTime(&elapsed, start, end));
        int repeats = std::clamp(int(std::ceil(50.0 / std::max(double(elapsed) / 3, .001))), 3, 10000);
        checked(cudaEventRecord(start));
        for (int i = 0; i < repeats; ++i) launch(route);
        checked(cudaEventRecord(end)); checked(cudaEventSynchronize(end));
        checked(cudaEventElapsedTime(&elapsed, start, end));
        std::printf("{\"k\":%d,\"m\":%d,\"route\":\"%s\",\"pass\":%d,\"repeats\":%d,\"stream_us\":%.6f,\"global_fixture_bytes\":%zu}\n",
            k, m, names[route], pass, repeats, elapsed * 1000.0 / repeats, weight_bytes + input.size() + output_bytes);
    }
    checked(cudaEventDestroy(start)); checked(cudaEventDestroy(end));
}
int positive(const char * value) {
    int result = 0; const char * end = value + std::strlen(value);
    auto parsed = std::from_chars(value, end, result);
    require(parsed.ec == std::errc{} && parsed.ptr == end && result > 0 && value[0] != '+' && value[0] != '-', "Invalid positive integer");
    return result;
}
}

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    require(!std::getenv("TS_GGML_Q8_PARALLEL_VECTOR") || std::strcmp(std::getenv("TS_GGML_Q8_PARALLEL_VECTOR"), "0") == 0,
        "Control requires parallel vector disabled");
    bool check = argc == 2 && std::string(argv[1]) == "--check";
    bool benchmark = argc == 4 && std::string(argv[1]) == "--benchmark";
    require(check || benchmark, "Use --check or --benchmark K M");
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    checked(cudaSetDevice(0));
    if (check) for (int k : {32, 96, 1024}) for (int m : {1, 3, 31, 32, 33, 63, 64, 65, 127, 129})
        for (bool padded : {false, true}) measure(k, m, padded, false);
    if (benchmark) {
        int k = positive(argv[2]), m = positive(argv[3]);
        require(k <= 16384 && k % 32 == 0 && m <= 262144, "Invalid bounded shape");
        measure(k, m, false, true);
    }
    return 0;
}
