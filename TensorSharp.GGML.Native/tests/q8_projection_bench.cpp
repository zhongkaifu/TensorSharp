// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Opt-in projection microcontrol. This is not an automatic CTest benchmark.
#include "ggml_ops_q8_precision.h"
#include "ggml_ops_matmul_precision.h"
#include "ggml_ops_dsv4_fused.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include "ggml-backend-impl.h"
#include <cuda_runtime_api.h>
#include "precision_test_utils.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace {
void check_cuda(cudaError_t status, const char * expression) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA failure in %s: %s\n", expression, cudaGetErrorString(status));
        std::exit(1);
    }
}
#define BENCH_CUDA(expression) check_cuda((expression), #expression)
enum class route { upstream, simt, expanded };
const char * name(route r) {
    return r == route::upstream ? "upstream_q8" : r == route::simt ? "owned_simt" : "transient_f32_cublas";
}

struct fixture {
    int k, m, n;
    ggml_context * ctx;
    ggml_tensor * w;
    ggml_tensor * w2;
    ggml_tensor * x;
    ggml_backend_buffer_t buffer;
    std::vector<unsigned char> weight_bytes, second_bytes;
    std::vector<float> weights, second, input;

    fixture(ggml_backend_t backend, int inner, int rows, int columns, bool second_weights = false) : k(inner), m(rows), n(columns) {
        ctx = ggml_init({1024 * 1024, nullptr, true});
        require(ctx != nullptr, "Cannot allocate Q8 microcontrol input context");
        w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, m);
        w2 = second_weights ? ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, m) : nullptr;
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer != nullptr, "Cannot allocate Q8 microcontrol inputs");
        weights.resize(size_t(k) * m);
        if (w2) second.resize(weights.size());
        input.resize(size_t(k) * n);
        weight_bytes.resize(ggml_nbytes(w));
        if (w2) second_bytes.resize(weight_bytes.size());
    }

    ~fixture() { ggml_backend_buffer_free(buffer); ggml_free(ctx); }

    void fill_weights(std::vector<unsigned char> & bytes, std::vector<float> & values, int revision, int exact) {
        require(ggml_type_size(GGML_TYPE_Q8_0) == 34 && ggml_blck_size(GGML_TYPE_Q8_0) == 32, "Q8 layout changed");
        for (int row = 0; row < m; ++row) for (int block = 0; block < k / 32; ++block) {
            // Independently construct IEEE half bits and the mathematical
            // scale. Nonzero mantissas ensure decoded weights need >F16 bits.
            const int exponent = 5 + (row + block) % 4;
            const int mantissa = 1 + (row * 71 + block * 19 + revision * 13) % 1023;
            const ggml_fp16_t bits = exact == 4 ? ggml_fp16_t(1) : exact == 1 ? ggml_fp16_t(0x3c01) : exact ? ggml_fp16_t(0x3c00)
                : ggml_fp16_t((exponent << 10) | mantissa);
            const double scale = exact == 4 ? std::ldexp(1.0, -24) : exact == 1 ? 1.0 + std::ldexp(1.0, -10) : exact ? 1.0
                : std::ldexp(1.0 + double(mantissa) / 1024.0, exponent - 15);
            auto * dst = bytes.data() + (size_t(row) * (k / 32) + block) * 34;
            std::memcpy(dst, &bits, 2);
            for (int j = 0; j < 32; ++j) {
                const int index = block * 32 + j;
                const int quant = exact == 1 ? (index == 0 ? 127 : 0)
                    : exact ? (index == 0 ? 1 : index == 1 ? -1 : 0)
                    : ((index * 37 + row * 19 + revision * 11) % 256) - 128;
                const int8_t encoded = int8_t(quant);
                std::memcpy(dst + 2 + j, &encoded, 1);
                values[size_t(row) * k + index] = float(scale * quant);
            }
        }
    }

    void upload(int revision = 0, int exact = 0) {
        fill_weights(weight_bytes, weights, revision, exact);
        if (w2) fill_weights(second_bytes, second, revision + 101, exact);
        for (int col = 0; col < n; ++col) for (int index = 0; index < k; ++index) {
            const uint32_t hash = uint32_t(index + 1) * 2654435761u ^ uint32_t(col + 102) * 2246822519u;
            float value = float(int(hash % 65537) - 32768) / 32768.0f + float((hash >> 17) % 13) * 0x1p-22f;
            if (exact == 1) value = index == 0 ? 1.0f : 0.0f;
            if (exact == 2) value = index == 0 ? 1.0f + 0x1p-16f : index == 1 ? 1.0f : 0.0f;
            if (exact == 3) value = index == 0 ? 70000.125f : index == 1 ? 70000.0f : 0.0f;
            if (exact == 4) value = index == 0 ? 0x1p-110f : 0.0f;
            input[size_t(col) * k + index] = revision == 0 ? value : value * 0.5f;
        }
        ggml_backend_tensor_set(w, weight_bytes.data(), 0, weight_bytes.size());
        if (w2) ggml_backend_tensor_set(w2, second_bytes.data(), 0, second_bytes.size());
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    }

    void check_unchanged() const {
        std::vector<unsigned char> bytes(weight_bytes.size());
        ggml_backend_tensor_get(w, bytes.data(), 0, bytes.size());
        require(bytes == weight_bytes, "Q8 microcontrol modified source weights");
        if (w2) {
            ggml_backend_tensor_get(w2, bytes.data(), 0, bytes.size());
            require(bytes == second_bytes, "Q8 microcontrol modified second source weights");
        }
        std::vector<float> actual(input.size());
        ggml_backend_tensor_get(x, actual.data(), 0, actual.size() * sizeof(float));
        require(std::memcmp(actual.data(), input.data(), actual.size() * sizeof(float)) == 0,
                "Q8 microcontrol modified activations");
    }
};

struct graph {
    ggml_context * ctx;
    ggml_cgraph * gf;
    ggml_gallocr_t allocator;
    ggml_tensor * output;
    ggml_tensor * expanded = nullptr;
    ggml_tensor * expanded_second = nullptr;
    ggml_backend_t backend;

    graph(fixture & f, route r, ggml_backend_t raw, ggml_backend_t wrapped, bool reuse = false) {
        ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
        require(ctx != nullptr, "Cannot allocate Q8 microcontrol graph context");
        backend = r == route::upstream ? raw : wrapped;
        if (r == route::upstream) output = ggml_mul_mat(ctx, f.w, f.x);
        else if (r == route::simt) output = tsg_matmul_q8_f32(ctx, f.w, f.x);
        else {
            expanded = ggml_cast(ctx, f.w, GGML_TYPE_F32);
            output = tsg_matmul_f32(ctx, expanded, f.x);
            if (reuse) {
                require(f.w2 != nullptr, "Missing second reuse-fixture weight matrix");
                expanded_second = ggml_cast(ctx, f.w2, GGML_TYPE_F32);
                auto * other = tsg_matmul_f32(ctx, expanded_second, f.x);
                output = ggml_add(ctx, output, other);
            }
        }
        ggml_set_output(output);
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, output);
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(raw));
        require(allocator && ggml_gallocr_alloc_graph(allocator, gf), "Cannot allocate Q8 microcontrol graph tensors");
        if (reuse) require(expanded->data == expanded_second->data,
                           "Two transient weight expansions did not reuse the same allocated storage");
    }

    ~graph() {
        ggml_backend_synchronize(backend);
        ggml_gallocr_free(allocator);
        ggml_free(ctx);
    }

    void compute(bool sync = true) {
        require(ggml_backend_graph_compute_async(backend, gf) == GGML_STATUS_SUCCESS, "Q8 microcontrol graph failed");
        if (sync) ggml_backend_synchronize(backend);
    }
};

struct oracle_result { size_t samples = 0, failures = 0; double maximum = 0, relative = 0; };
std::vector<int> indices(int count) {
    std::set<int> result{0, count - 1, count / 2};
    for (int value : {1, 7, 16, 31, 63, 127, 151, 421, 575, 1223}) if (value < count) result.insert(value);
    return {result.begin(), result.end()};
}

oracle_result validate(const fixture & f, const graph & g, bool strict, bool reuse = false, int exact = 0) {
    std::vector<float> actual(size_t(f.m) * f.n);
    ggml_backend_tensor_get(g.output, actual.data(), 0, actual.size() * sizeof(float));
    for (float value : actual) require(std::isfinite(value), "Q8 microcontrol produced a nonfinite output");
    oracle_result result;
    double squared_error = 0, squared_reference = 0;
    auto rows = indices(f.m), columns = indices(f.n);
    if (exact) {
        rows.resize(f.m); columns.resize(f.n);
        std::iota(rows.begin(), rows.end(), 0); std::iota(columns.begin(), columns.end(), 0);
    }
    for (int col : columns) for (int row : rows) {
        double expected = 0;
        for (int index = 0; index < f.k; ++index) {
            const size_t wi = size_t(row) * f.k + index;
            expected += double(f.weights[wi]) * f.input[size_t(col) * f.k + index];
            if (reuse) expected += double(f.second[wi]) * f.input[size_t(col) * f.k + index];
        }
        if (exact == 1) require(expected == 127.1240234375, "Exact decoded-weight fixture changed");
        const double error = std::abs(double(actual[size_t(col) * f.m + row]) - expected);
        const double tolerance = exact ? 0.0 : 0.0001 + 0.000006 * std::abs(expected);
        if (error > tolerance) ++result.failures;
        result.maximum = std::max(result.maximum, error);
        squared_error += error * error;
        squared_reference += expected * expected;
        ++result.samples;
    }
    result.relative = std::sqrt(squared_error / std::max(squared_reference, 1e-300));
    if (strict) require(result.failures == 0 && result.relative <= 0.000004,
                        "Q8 microcontrol failed independent sampled double oracle");
    else require(result.relative < 0.02, "Upstream Q8 microcontrol error exceeded its approximate baseline bound");
    f.check_unchanged();
    return result;
}

void guards(ggml_backend_t raw, ggml_backend_t wrapped) {
    for (int exact : {1, 2, 3, 4}) for (int columns : {1, 17}) {
        fixture f(raw, 64, 65, columns);
        f.upload(0, exact);
        graph g(f, route::expanded, raw, wrapped);
        g.compute();
        const auto result = validate(f, g, true, false, exact);
        std::printf("{\"guard\":\"exact_expanded\",\"pattern\":%d,\"n\":%d,\"samples\":%zu,\"max_error\":%.9g}\n",
                    exact, columns, result.samples, result.maximum);
    }
    fixture f(raw, 384, 1536, 17, true);
    graph g(f, route::expanded, raw, wrapped, true);
    for (int revision = 0; revision < 3; ++revision) {
        f.upload(revision);
        g.compute();
        const auto result = validate(f, g, true, true);
        std::printf("{\"guard\":\"transient_storage_reuse\",\"revision\":%d,\"same_storage\":true,\"graph_bytes\":%zu,\"one_expansion_bytes\":%zu,\"samples\":%zu,\"max_error\":%.9g}\n",
                    revision, ggml_gallocr_get_buffer_size(g.allocator, 0), ggml_nbytes(g.expanded), result.samples, result.maximum);
    }
}

void measure(fixture & f, route r, ggml_backend_t raw, ggml_backend_t wrapped, int pass, double minimum_ms) {
    size_t free_before, total, free_allocated, free_warm;
    BENCH_CUDA(cudaMemGetInfo(&free_before, &total));
    graph g(f, r, raw, wrapped);
    BENCH_CUDA(cudaMemGetInfo(&free_allocated, &total));
    for (int repeat = 0; repeat < 3; ++repeat) g.compute();
    const auto oracle = validate(f, g, r != route::upstream);
    BENCH_CUDA(cudaMemGetInfo(&free_warm, &total));
    const auto pilot_start = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < 10; ++repeat) g.compute(false);
    ggml_backend_synchronize(g.backend);
    const double pilot_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pilot_start).count() / 10;
    const int repeats = std::max(3, std::min(10000, int(std::ceil(minimum_ms / std::max(pilot_ms, 0.001)))));
    cudaEvent_t begin, end;
    BENCH_CUDA(cudaEventCreate(&begin)); BENCH_CUDA(cudaEventCreate(&end));
    // Owned timing-enabled CUDA events use the unchanged backend's event
    // record interface, avoiding private CUDA context/layout dependencies.
    ggml_backend_event begin_event{ggml_backend_get_device(raw), begin};
    ggml_backend_event end_event{ggml_backend_get_device(raw), end};
    const auto wall_start = std::chrono::steady_clock::now();
    ggml_backend_event_record(&begin_event, raw);
    for (int repeat = 0; repeat < repeats; ++repeat) g.compute(false);
    ggml_backend_event_record(&end_event, raw);
    BENCH_CUDA(cudaEventSynchronize(end));
    const double wall_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - wall_start).count() / repeats;
    float elapsed_ms;
    BENCH_CUDA(cudaEventElapsedTime(&elapsed_ms, begin, end));
    BENCH_CUDA(cudaEventDestroy(begin)); BENCH_CUDA(cudaEventDestroy(end));
    validate(f, g, r != route::upstream);
    std::printf("{\"route\":\"%s\",\"k\":%d,\"m\":%d,\"n\":%d,\"pass\":%d,\"repeats\":%d,\"wall_us\":%.6f,\"stream_us\":%.6f,\"graph_allocated_bytes\":%zu,\"leaf_allocated_bytes\":%zu,\"expansion_bytes\":%zu,\"observed_graph_device_delta\":%lld,\"observed_warm_device_delta\":%lld,\"oracle_samples\":%zu,\"output_elements\":%zu,\"oracle_max_error\":%.9g,\"oracle_relative_l2\":%.9g,\"strict_oracle_failures\":%zu,\"strict_required\":%s}\n",
        name(r), f.k, f.m, f.n, pass, repeats, wall_us, double(elapsed_ms) * 1000 / repeats,
        ggml_gallocr_get_buffer_size(g.allocator, 0), ggml_backend_buffer_get_size(f.buffer),
        g.expanded ? ggml_nbytes(g.expanded) : 0, static_cast<long long>(free_before) - static_cast<long long>(free_allocated),
        static_cast<long long>(free_before) - static_cast<long long>(free_warm), oracle.samples, size_t(f.m) * f.n,
        oracle.maximum, oracle.relative, oracle.failures, r == route::upstream ? "false" : "true");
}
} // namespace

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    bool benchmark = false;
    int device = 0, k = 0, m = 0, n = 0;
    double minimum_ms = 100;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--benchmark") benchmark = true;
        else if (arg == "--device" && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (arg == "--min-ms" && i + 1 < argc) minimum_ms = std::atof(argv[++i]);
        else if (arg == "--shape" && i + 3 < argc) { k = std::atoi(argv[++i]); m = std::atoi(argv[++i]); n = std::atoi(argv[++i]); }
        else { std::fprintf(stderr, "Unknown/incomplete argument: %s\n", argv[i]); return 2; }
    }
    if (!benchmark) {
        std::fprintf(stderr, "Opt-in only: --benchmark [--device 0] [--min-ms 100] [--shape K M N]\n");
        return 2;
    }
    require(minimum_ms > 0 && minimum_ms <= 10000, "Invalid timing duration");
    if (k || m || n) require(k > 0 && k % 32 == 0 && k <= 8192 && m > 0 && m <= 8192 && n > 0 && n <= 8192,
                             "Invalid bounded benchmark shape");
    const int count = ggml_backend_cuda_get_device_count();
    if (count == 0) return 77;
    require(device >= 0 && device < count, "Invalid benchmark device");
    auto * raw = ggml_backend_cuda_init(device);
    require(raw != nullptr, "Cannot initialize microcontrol CUDA backend");
    auto * wrapped = tsg_dsv4_fused_backend_init(raw);
    require(wrapped != nullptr, "Cannot initialize microcontrol owned backend");
    guards(raw, wrapped);
    std::vector<std::array<int, 3>> shapes;
    if (k) shapes.push_back({k, m, n});
    else for (const auto dims : {std::array<int, 2>{384, 1152}, {384, 384}, {384, 1536}, {1536, 384},
                                 {1024, 3072}, {1024, 1024}, {1024, 4096}, {4096, 1024}})
        for (int columns : {17, 152, 422, 144, 1224, 576}) shapes.push_back({dims[0], dims[1], columns});
    for (const auto shape : shapes) {
        fixture f(raw, shape[0], shape[1], shape[2]);
        f.upload();
        // Reversed second pass limits a simple warm-clock/order advantage.
        // This is a projection microcontrol, not a full-model ABBA campaign.
        for (int pass = 0; pass < 2; ++pass) {
            const route routes[] = {route::upstream, route::simt, route::expanded};
            for (int i = 0; i < 3; ++i) measure(f, routes[pass == 0 ? i : 2 - i], raw, wrapped, pass, minimum_ms);
        }
    }
    ggml_backend_free(wrapped); ggml_backend_free(raw);
    return 0;
}
