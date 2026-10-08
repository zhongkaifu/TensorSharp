// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Opt-in exploration: bounded Q8->F32 row tiles + pedantic F32 GEMM.
// This executable never changes model or production projection dispatch.
#include "ggml_ops_q8_precision.h"
#include "ggml_ops_shared_cache_budget.h"
#include "precision_test_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <string>

namespace {
void checked(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }
void checked(cublasStatus_t status) { require(status == CUBLAS_STATUS_SUCCESS, "Pedantic cuBLAS operation failed"); }
constexpr size_t WorkspaceBytes = 4 * 1024 * 1024;
size_t aligned(size_t bytes) { require(bytes <= SIZE_MAX - 255, "Scratch alignment overflow"); return (bytes + 255) / 256 * 256; }

struct ScratchBudget {
    size_t capacity, live = 0, peak = 0;
    int allocations = 0;
    static uint64_t reserve(void * context, int rank, int kind, int64_t bytes) {
        auto & value = *static_cast<ScratchBudget *>(context);
        require(rank == 0 && kind == 2 && bytes > 0, "Unexpected scratch admission request");
        if (size_t(bytes) > value.capacity - value.live) return 0;
        value.live += size_t(bytes); value.peak = std::max(value.peak, value.live);
        return uint64_t(bytes); // one arena per isolated benchmark
    }
    static int commit(void * context, uint64_t bytes) {
        return static_cast<ScratchBudget *>(context)->live == size_t(bytes) ? 1 : 0;
    }
    static void release(void * context, uint64_t bytes) {
        auto & value = *static_cast<ScratchBudget *>(context);
        require(value.live == size_t(bytes), "Scratch release did not match its owner");
        value.live -= size_t(bytes);
    }
};

struct DeviceAllocation {
    void * data = nullptr;
    explicit DeviceAllocation(size_t bytes) { checked(cudaMalloc(&data, bytes)); }
    ~DeviceAllocation() { if (data) checked(cudaFree(data)); }
    DeviceAllocation(const DeviceAllocation &) = delete;
};

__global__ void dequantize_rows(const char * source, float * destination, int k, int rows, size_t stride) {
    const int64_t count = int64_t(k) * rows;
    for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += int64_t(blockDim.x) * gridDim.x) {
        const int row = int(index / k), inner = int(index % k);
        const char * block = source + size_t(row) * stride + size_t(inner / 32) * 34;
        const float scale = __half2float(*reinterpret_cast<const half *>(block));
        destination[index] = scale * float(*reinterpret_cast<const int8_t *>(block + 2 + inner % 32));
    }
}

__global__ void pack_input(const char * source, float * destination, int k, int columns,
                          size_t inner_stride, size_t column_stride) {
    const int64_t count = int64_t(k) * columns;
    for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += int64_t(blockDim.x) * gridDim.x)
        destination[index] = *reinterpret_cast<const float *>(source + size_t(index / k) * column_stride
                                                             + size_t(index % k) * inner_stride);
}

unsigned grid(size_t count) { return unsigned(std::min<size_t>((count + 255) / 256, 65535)); }

struct TiledGemm {
    int inner, row_capacity, column_capacity;
    size_t bytes, weight_offset, input_offset;
    bool packed_input;
    void * arena = nullptr;
    cublasHandle_t handle = nullptr;
    std::shared_ptr<tsg::SharedCacheCharge> charge;

    static size_t payload(int k, int rows, int columns, bool pack) {
        return WorkspaceBytes + aligned(size_t(k) * rows * sizeof(float))
            + (pack ? aligned(size_t(k) * columns * sizeof(float)) : 0);
    }
    TiledGemm(ScratchBudget & budget, int k, int rows, int columns, bool pack)
        : inner(k), row_capacity(rows), column_capacity(columns), bytes(payload(k, rows, columns, pack)),
          weight_offset(WorkspaceBytes), input_offset(WorkspaceBytes + aligned(size_t(k) * rows * sizeof(float))),
          packed_input(pack) {
        charge = tsg::SharedCacheCharge::reserve(0, 2, bytes);
        if (!charge) return; // no physical payload exists if quota denied
        checked(cudaMalloc(&arena, bytes)); ++budget.allocations;
        require(charge->commit(bytes), "Scratch commit refused after physical allocation");
        checked(cublasCreate(&handle));
        checked(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
        checked(cublasSetStream(handle, nullptr));
        checked(cublasSetWorkspace(handle, arena, WorkspaceBytes));
    }
    ~TiledGemm() {
        if (!arena) return;
        checked(cudaStreamSynchronize(nullptr));
        if (handle) checked(cublasDestroy(handle));
        checked(cudaFree(arena));
        charge.reset(); // credit returns only after physical payload release
    }
    TiledGemm(const TiledGemm &) = delete;

    void project(const void * weights, const void * input, float * output, int rows, int columns,
                 size_t weight_stride, size_t input_inner_stride, size_t input_column_stride) {
        require(arena && handle, "Attempted execution without admitted workspace");
        auto * widened = reinterpret_cast<float *>(static_cast<char *>(arena) + weight_offset);
        auto * packed = reinterpret_cast<float *>(static_cast<char *>(arena) + input_offset);
        const float alpha = 1, beta = 0;
        for (int first_row = 0; first_row < rows; first_row += row_capacity) {
            const int count_rows = std::min(row_capacity, rows - first_row);
            dequantize_rows<<<grid(size_t(inner) * count_rows), 256>>>(
                static_cast<const char *>(weights) + size_t(first_row) * weight_stride,
                widened, inner, count_rows, weight_stride);
            checked(cudaGetLastError());
            for (int first_column = 0; first_column < columns; first_column += column_capacity) {
                const int count_columns = std::min(column_capacity, columns - first_column);
                const char * source = static_cast<const char *>(input) + size_t(first_column) * input_column_stride;
                int leading_input = int(input_column_stride / sizeof(float));
                if (packed_input) {
                    pack_input<<<grid(size_t(inner) * count_columns), 256>>>(source, packed,
                        inner, count_columns, input_inner_stride, input_column_stride);
                    checked(cudaGetLastError());
                    source = reinterpret_cast<const char *>(packed); leading_input = inner;
                }
                // Complete K for each output: no split-K partial sums. ldc is
                // the full destination row count, so no output tile is needed.
                checked(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    count_rows, count_columns, inner, &alpha, widened, CUDA_R_32F, inner,
                    source, CUDA_R_32F, leading_input, &beta,
                    output + size_t(first_column) * rows + first_row, CUDA_R_32F, rows,
                    CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
                checked(cudaGetLastError());
            }
        }
    }
};

struct Fixture {
    int k, m, n;
    size_t ws, xs, cs;
    bool strided;
    std::vector<unsigned char> weights;
    std::vector<float> input;
    Fixture(int inner, int rows, int columns, bool padded) : k(inner), m(rows), n(columns),
        ws(size_t(k / 32 + (padded ? 3 : 0)) * 34), xs((padded ? 2 : 1) * sizeof(float)),
        cs((size_t(k) * (padded ? 2 : 1) + (padded ? 3 : 0)) * sizeof(float)), strided(padded),
        weights(size_t(m) * ws, 0xff), input(size_t(n) * cs / sizeof(float), std::nanf("")) {
        for (int row = 0; row < m; ++row) for (int block = 0; block < k / 32; ++block) {
            const uint16_t scale_bits = uint16_t(((8 + (row + block) % 4) << 10) | ((row * 71 + block * 19) % 1024));
            auto * destination = weights.data() + size_t(row) * ws + size_t(block) * 34;
            std::memcpy(destination, &scale_bits, 2);
            for (int j = 0; j < 32; ++j) {
                const int8_t value = int8_t(quant(row, block * 32 + j));
                std::memcpy(destination + 2 + j, &value, 1);
            }
        }
        for (int column = 0; column < n; ++column) for (int inner = 0; inner < k; ++inner)
            input[(size_t(column) * cs + size_t(inner) * xs) / sizeof(float)] = activation(inner, column);
    }
    static int quant(int row, int inner) { return (inner * 37 + row * 19) % 256 - 128; }
    static double weight(int row, int inner) {
        const int block = inner / 32;
        const double scale = std::ldexp(1.0 + double((row * 71 + block * 19) % 1024) / 1024.0, -7 + (row + block) % 4);
        return scale * quant(row, inner);
    }
    static float activation(int inner, int column) {
        const uint32_t hash = uint32_t(inner + 1) * 2654435761u ^ uint32_t(column + 101) * 2246822519u;
        return float(int(hash % 65537) - 32768) / 32768.0f + float((hash >> 17) % 13) * 0x1p-22f;
    }
};

struct Metrics { double relative = 0, maximum = 0; size_t samples = 0, failed_samples = 0; bool passed = true; };
Metrics verify(const Fixture & fixture, const std::vector<float> & output, bool enforce_oracle, bool full_oracle) {
    require(output.front() == -12345.625f && output.back() == -12345.625f, "Prefill output canary changed");
    for (size_t i = 1; i + 1 < output.size(); ++i) require(std::isfinite(output[i]), "Nonfinite prefill output");
    std::set<int> rows{0, fixture.m - 1}, columns{0, fixture.n - 1};
    if (full_oracle) {
        for (int row = 0; row < fixture.m; ++row) rows.insert(row);
        for (int column = 0; column < fixture.n; ++column) columns.insert(column);
    } else {
        for (int i = 0; i < 31; ++i) rows.insert(int(int64_t(i) * (fixture.m - 1) / 30));
        for (int i = 0; i < 17; ++i) columns.insert(int(int64_t(i) * (fixture.n - 1) / 16));
    }
    double errors = 0, norm = 0; Metrics metrics;
    for (int column : columns) for (int row : rows) {
        double expected = 0, sum_absolute = 0;
        for (int k = 0; k < fixture.k; ++k) {
            const double product = Fixture::weight(row, k) * Fixture::activation(k, column);
            expected += product; sum_absolute += std::abs(product);
        }
        const double error = std::abs(double(output[1 + size_t(column) * fixture.m + row]) - expected);
        if (enforce_oracle) {
            if (error > 0.0001 + 0.000006 * std::abs(expected)) {
                const double unit = std::ldexp(1.0, -24), gamma = fixture.k * unit / (1 - fixture.k * unit);
                std::fprintf(stderr,
                    "FP64 diagnostic: K=%d M=%d N=%d row=%d column=%d expected=%.17g actual=%.17g error=%.17g original_gate=%.17g sum_absolute=%.17g gamma_k_bound=%.17g\n",
                    fixture.k, fixture.m, fixture.n, row, column, expected,
                    double(output[1 + size_t(column) * fixture.m + row]), error,
                    0.0001 + 0.000006 * std::abs(expected), sum_absolute, gamma * sum_absolute);
            }
            if (error > 0.0001 + 0.000006 * std::abs(expected)) ++metrics.failed_samples;
        }
        errors += error * error; norm += expected * expected;
        metrics.maximum = std::max(metrics.maximum, error); ++metrics.samples;
    }
    metrics.relative = std::sqrt(errors / std::max(norm, std::numeric_limits<double>::min()));
    metrics.passed = metrics.failed_samples == 0 && metrics.relative <= 0.000004;
    return metrics;
}

template<class Launch> double timing(Launch launch) {
    cudaEvent_t first, last; checked(cudaEventCreate(&first)); checked(cudaEventCreate(&last));
    launch(); checked(cudaDeviceSynchronize());
    checked(cudaEventRecord(first)); launch(); checked(cudaEventRecord(last)); checked(cudaEventSynchronize(last));
    float elapsed; checked(cudaEventElapsedTime(&elapsed, first, last));
    const int repetitions = std::clamp(int(std::ceil(30.0 / std::max(double(elapsed), 0.01))), 2, 1000);
    checked(cudaEventRecord(first));
    for (int i = 0; i < repetitions; ++i) launch();
    checked(cudaEventRecord(last)); checked(cudaEventSynchronize(last));
    checked(cudaEventElapsedTime(&elapsed, first, last));
    checked(cudaEventDestroy(first)); checked(cudaEventDestroy(last));
    return double(elapsed) * 1000 / repetitions;
}

bool measure(int k, int m, int n, bool padded, size_t ceiling, bool benchmark) {
    Fixture fixture(k, m, n, padded);
    DeviceAllocation weights(fixture.weights.size()), input(fixture.input.size() * sizeof(float));
    const size_t output_count = size_t(m) * n + 2;
    DeviceAllocation serial(output_count * sizeof(float)), candidate(output_count * sizeof(float));
    checked(cudaMemcpy(weights.data, fixture.weights.data(), fixture.weights.size(), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(input.data, fixture.input.data(), fixture.input.size() * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<float> poison(output_count, std::nanf(""));
    poison.front() = poison.back() = -12345.625f;
    std::vector<float> result = poison;
    checked(cudaMemcpy(serial.data, result.data(), result.size() * sizeof(float), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(candidate.data, result.data(), result.size() * sizeof(float), cudaMemcpyHostToDevice));
    auto old = [&] { checked(static_cast<cudaError_t>(tsg_matmul_q8_cuda_launch(weights.data, input.data,
        static_cast<float *>(serial.data) + 1, k, m, n, fixture.ws, fixture.xs, fixture.cs, nullptr))); };
    old(); checked(cudaDeviceSynchronize());
    checked(cudaMemcpy(result.data(), serial.data, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    // The existing serial kernel is a measured control, not the truth oracle.
    // Report its error separately; candidate acceptance uses the FP64 gate.
    const Metrics old_error = verify(fixture, result, true, !benchmark);
    const double before_us = benchmark ? timing(old) : 0;
    ScratchBudget budget{ceiling};
    require(tsg::SharedCacheCharge::attach(&budget, ScratchBudget::reserve, ScratchBudget::commit, ScratchBudget::release, true),
        "Cannot attach isolated scratch ledger");
    double best_us = std::numeric_limits<double>::infinity(); int best_rows = 0, best_columns = 0;
    bool all_qualified = true;
    std::set<int> row_tiles, column_tiles;
    for (int rows : (benchmark ? std::vector<int>{64, 256, 1024, 4096} : std::vector<int>{1, 63, 128}))
        row_tiles.insert(std::min(rows, m));
    for (int columns : (benchmark ? std::vector<int>{32, 128, n} : std::vector<int>{1, 8, n}))
        column_tiles.insert(std::min(columns, n));
    for (int rows : row_tiles) for (int columns : column_tiles) {
        const int allocations_before = budget.allocations;
        {
            TiledGemm projection(budget, k, rows, columns, padded);
            if (!projection.arena) {
                require(budget.live == 0 && budget.allocations == allocations_before, "Refused scratch still allocated payload");
                std::printf("{\"route\":\"pedantic_f32_gemm\",\"admitted\":false,\"k\":%d,\"m\":%d,\"n\":%d,\"tile_rows\":%d,\"tile_columns\":%d,\"requested_bytes\":%zu,\"scratch_cap_bytes\":%zu}\n",
                    k, m, n, rows, columns, projection.bytes, ceiling);
                continue;
            }
            auto project = [&] { projection.project(weights.data, input.data, static_cast<float *>(candidate.data) + 1,
                m, n, fixture.ws, fixture.xs, fixture.cs); };
            checked(cudaMemcpy(candidate.data, poison.data(), poison.size() * sizeof(float), cudaMemcpyHostToDevice));
            project(); checked(cudaDeviceSynchronize());
            checked(cudaMemcpy(result.data(), candidate.data, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
            const Metrics error = verify(fixture, result, true, !benchmark);
            const double first_us = benchmark ? timing(project) : 0, second_us = benchmark ? timing(project) : 0;
            const double average = (first_us + second_us) / 2;
            all_qualified = all_qualified && error.passed;
            if (error.passed && average < best_us) { best_us = average; best_rows = rows; best_columns = columns; }
            std::printf("{\"oracle_qualified\":%s,\"failed_samples\":%zu,\"tile_rows\":%d,\"tile_columns\":%d}\n",
                error.passed ? "true" : "false", error.failed_samples, rows, columns);
            std::printf("{\"data\":\"synthetic\",\"production_candidate\":false,\"route\":\"pedantic_f32_gemm\",\"admitted\":true,\"k\":%d,\"m\":%d,\"n\":%d,\"strided\":%s,\"tile_rows\":%d,\"tile_columns\":%d,\"scratch_cap_bytes\":%zu,\"scratch_payload_bytes\":%zu,\"workspace_bytes\":%zu,\"stream_us_1\":%.6f,\"stream_us_2\":%.6f,\"oracle_samples\":%zu,\"oracle_max_error\":%.9g,\"oracle_relative_l2\":%.9g}\n",
                k, m, n, padded ? "true" : "false", rows, columns, ceiling, projection.bytes, WorkspaceBytes,
                first_us, second_us, error.samples, error.maximum, error.relative);
        }
        require(budget.live == 0, "Scratch credit survived physical owner cleanup");
    }
    require(tsg::SharedCacheCharge::detach(&budget), "Scratch scope could not detach");
    const double after_us = benchmark ? timing(old) : 0;
    std::printf("{\"route\":\"serial_f32_control\",\"k\":%d,\"m\":%d,\"n\":%d,\"stream_us_before\":%.6f,\"stream_us_after\":%.6f,\"oracle_relative_l2\":%.9g,\"oracle_max_error\":%.9g,\"oracle_qualified\":%s,\"failed_samples\":%zu,\"best_observed_tile_rows\":%d,\"best_observed_tile_columns\":%d,\"best_observed_stream_us\":%.6f,\"scratch_peak_bytes\":%zu,\"baseline_payload_bytes\":%zu,\"library_metadata_excluded\":true,\"final_scratch_live_bytes\":%zu}\n",
        k, m, n, before_us, after_us, old_error.relative, old_error.maximum, old_error.passed ? "true" : "false", old_error.failed_samples, best_rows, best_columns,
        std::isfinite(best_us) ? best_us : 0, budget.peak,
        fixture.weights.size() + fixture.input.size() * sizeof(float) + 2 * output_count * sizeof(float), budget.live);
    return all_qualified && old_error.passed;
}
}

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc != 2 && argc != 6) { std::fprintf(stderr, "Use --check or --benchmark [K M N SCRATCH_MiB].\n"); return 2; }
    const bool benchmark = std::string(argv[1]) == "--benchmark";
    require(benchmark || (argc == 2 && std::string(argv[1]) == "--check"), "Unknown benchmark mode");
    const char * experimental = std::getenv("TS_GGML_Q8_PARALLEL_VECTOR");
    require(!experimental || std::strcmp(experimental, "1") != 0, "Unset vector experiment for the serial control");
    experimental = std::getenv("TS_GGML_Q8_PARALLEL_SMALL_BATCH");
    require(!experimental || std::strcmp(experimental, "1") != 0, "Unset small-batch experiment for the serial control");
    int devices = 0; if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    checked(cudaSetDevice(0));
    bool passed = true;
    if (argc == 6) {
        const int k = std::atoi(argv[2]), m = std::atoi(argv[3]), n = std::atoi(argv[4]), mib = std::atoi(argv[5]);
        require(k > 0 && k <= 8192 && k % 32 == 0 && m > 0 && m <= 262144 && n > 1 && n <= 8192
            && mib > 0 && mib <= 512 && int64_t(m) * n <= 16 * 1024 * 1024, "Invalid bounded benchmark shape");
        passed = measure(k, m, n, false, size_t(mib) * 1024 * 1024, true);
    } else if (benchmark) {
        const int shapes[][2] = {{1024, 5120}, {1024, 6144}, {1024, 7168}, {2048, 1024}, {3584, 1024}};
        for (const auto & shape : shapes)
            for (size_t cap : {8u * 1024 * 1024, 32u * 1024 * 1024}) passed = measure(shape[0], shape[1], 643, false, cap, true) && passed;
    } else {
        for (int n : {2, 8, 9, 16, 17, 32, 33, 65}) passed = measure(96, 129, n, true, 8 * 1024 * 1024, false) && passed;
        // Same shape, one-byte-short cap must refuse every allocation.
        passed = measure(96, 129, 9, true, TiledGemm::payload(96, 1, 1, true) - 1, false) && passed;
        passed = measure(96, 129, 9, true, TiledGemm::payload(96, 1, 1, true), false) && passed;
    }
    return passed ? 0 : 1;
}
