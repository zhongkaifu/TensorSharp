// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_q8_precision.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#include "ggml_ops_dsv4_fused.h"
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
struct options {
    int k = 0, rows = 0, columns = 0, device = 0, threads = 4;
    int warmup = 5, repeats = 10, iterations = 10, sample_rows = 16, sample_columns = 16;
    uint64_t oracle_max_operations = 300000000, max_payload_bytes = 1073741824;
    double max_abs_gate = 5e-5, relative_l2_gate = 5e-6;
    std::string backend = "cuda", oracle = "sample", output;
};
uint64_t integer(const std::string& value) {
    require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
        "Expected an unsigned decimal integer: " + value);
    size_t used = 0;
    const auto result = std::stoull(value, &used);
    require(used == value.size(), "Invalid integer");
    return result;
}
options parse(int argc, char** argv) {
    options o;
    for (int index = 1; index < argc; index += 2) {
        require(index + 1 < argc, "Every option requires a value");
        const std::string key(argv[index]), value(argv[index + 1]);
        const auto as_int = [&]() { const auto n = integer(value); require(n <= 2147483647, "Integer exceeds INT_MAX"); return int(n); };
        if (key == "--k") o.k = as_int();
        else if (key == "--rows") o.rows = as_int();
        else if (key == "--columns") o.columns = as_int();
        else if (key == "--device") o.device = as_int();
        else if (key == "--threads") o.threads = as_int();
        else if (key == "--warmup") o.warmup = as_int();
        else if (key == "--repeats") o.repeats = as_int();
        else if (key == "--iterations") o.iterations = as_int();
        else if (key == "--sample-rows") o.sample_rows = as_int();
        else if (key == "--sample-columns") o.sample_columns = as_int();
        else if (key == "--oracle-max-operations") o.oracle_max_operations = integer(value);
        else if (key == "--max-payload-bytes") o.max_payload_bytes = integer(value);
        else if (key == "--backend") o.backend = value;
        else if (key == "--oracle") o.oracle = value;
        else if (key == "--output") o.output = value;
        else throw std::runtime_error("Unknown option: " + key);
    }
    require(o.k > 0 && o.k % 256 == 0 && o.rows > 0 && o.columns > 0, "K must be positive and divisible by 256; rows/columns must be positive");
    require(o.columns <= 524280, "Columns exceed the owned CUDA launch bound");
    require(o.backend == "cuda" || o.backend == "cpu", "Backend must be cuda or cpu");
    require(o.oracle == "sample" || o.oracle == "full" || o.oracle == "none", "Oracle must be sample, full or none");
    require(o.threads > 0 && o.warmup > 0 && o.repeats > 0 && o.iterations > 0,
        "Warmup/repeats/iterations/threads must be positive");
    require(o.warmup <= 10000 && o.repeats <= 10000 && o.iterations <= 100000,
        "Repeat count exceeds the bounded driver limit");
    require(o.sample_rows > 0 && o.sample_columns > 0 && o.oracle_max_operations > 0,
        "Oracle sampling counts and operation limit must be positive");
    require(!o.output.empty() && !std::filesystem::exists(o.output), "Output must be a fresh directory");
    require(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4, "IEEE binary32 is required");
    const uint16_t endian = 1;
    require(*reinterpret_cast<const unsigned char*>(&endian) == 1, "Little-endian binary output is required");
    const uint64_t weight = uint64_t(o.k / 256) * 176 * o.rows;
    const uint64_t source = uint64_t(o.k) * o.columns * 4;
    const uint64_t output = uint64_t(o.rows) * o.columns * 4;
    require(weight <= o.max_payload_bytes && source <= o.max_payload_bytes && output <= o.max_payload_bytes
        && weight + source + 2 * output <= o.max_payload_bytes,
        "Resident tensor payload exceeds the explicit per-case cap");
    if (o.oracle != "none") {
        const uint64_t oracle_rows = o.oracle == "full" ? o.rows : std::min(o.rows, o.sample_rows);
        const uint64_t oracle_columns = o.oracle == "full" ? o.columns : std::min(o.columns, o.sample_columns);
        require(oracle_rows * oracle_columns <= o.oracle_max_operations / uint64_t(o.k),
            "Requested F64 oracle exceeds --oracle-max-operations");
    }
    return o;
}
struct backends {
    ggml_backend_t original = nullptr, owned = nullptr;
    ~backends() { if (owned && owned != original) ggml_backend_free(owned); if (original) ggml_backend_free(original); }
};
struct storage {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~storage() { if (buffer) ggml_backend_buffer_free(buffer); if (ctx) ggml_free(ctx); }
};
uint32_t mix(uint32_t x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16); }
void fields(unsigned char* block, const std::array<unsigned, 8>& scales, const std::array<unsigned, 8>& minima) {
    for (unsigned group = 0; group < 8; ++group) for (unsigned bit = 0; bit < 6; ++bit) {
        const unsigned sb = group < 4 ? group : bit < 4 ? group + 4 : group - 4;
        const unsigned sp = group < 4 || bit < 4 ? bit : bit + 2;
        const unsigned mb = group < 4 ? group + 4 : bit < 4 ? group + 4 : group;
        const unsigned mp = group < 4 ? bit : bit < 4 ? bit + 4 : bit + 2;
        block[4 + sb] |= ((scales[group] >> bit) & 1) << sp;
        block[4 + mb] |= ((minima[group] >> bit) & 1) << mp;
    }
}
std::vector<unsigned char> make_weights(const options& o) {
    std::vector<unsigned char> bytes(size_t(o.k / 256) * 176 * o.rows, 0);
    for (int row = 0; row < o.rows; ++row) for (int b = 0; b < o.k / 256; ++b) {
        auto* p = bytes.data() + (size_t(row) * (o.k / 256) + b) * 176;
        // Exact power-of-two scales: independent stored-weight F64 oracle and
        // production decoder represent every generated weight exactly in F32.
        p[1] = 0x0c; p[3] = 0x08; // d=2^-12, d_min=2^-13.
        std::array<unsigned, 8> s{}, m{};
        for (unsigned g = 0; g < 8; ++g) {
            s[g] = 1 + mix(uint32_t(row) * 313u + uint32_t(b) * 17u + g) % 63;
            m[g] = mix(uint32_t(row) * 179u + uint32_t(b) * 13u + g + 71u) % 64;
        }
        fields(p, s, m);
        for (unsigned e = 0; e < 256; ++e) {
            const unsigned q = mix(uint32_t(row) * 65537u + uint32_t(b) * 257u + e + 29u) % 32;
            p[48 + (e / 64) * 32 + e % 32] |= (q & 15) << ((e / 32) % 2 ? 4 : 0);
            p[16 + e % 32] |= (q >> 4) << (e / 32);
        }
    }
    return bytes;
}
std::vector<float> make_input(const options& o) {
    std::vector<float> x(size_t(o.k) * o.columns);
    for (int c = 0; c < o.columns; ++c) for (int k = 0; k < o.k; ++k) {
        const int v = int(mix(uint32_t(k) * 15485863u + uint32_t(c) * 32452843u + 97u) % 2001) - 1000;
        x[size_t(c) * o.k + k] = std::ldexp(float(v), -13);
    }
    return x;
}
double weight(const unsigned char* p, int e) {
    const int group = e / 32, lane = e % 32;
    const auto* f = p + 4;
    const int scale = group < 4 ? f[group] & 63 : (f[group + 4] & 15) | ((f[group - 4] >> 6) << 4);
    const int minimum = group < 4 ? f[group + 4] & 63 : (f[group + 4] >> 4) | ((f[group] >> 6) << 4);
    const int q = ((p[48 + (group / 2) * 32 + lane] >> (group % 2 ? 4 : 0)) & 15)
        | (((p[16 + lane] >> group) & 1) << 4);
    return std::ldexp(double(scale * q), -12) - std::ldexp(double(minimum), -13);
}
std::vector<int> selected(int count, int limit) {
    std::vector<int> result;
    const int n = std::min(count, limit);
    for (int i = 0; i < n; ++i) result.push_back(n == 1 ? 0 : int(int64_t(i) * (count - 1) / (n - 1)));
    return result;
}
struct error { size_t count = 0; double maximum = 0, squared = 0, reference_squared = 0; };
std::array<error, 2> oracle(const options& o, const std::vector<unsigned char>& w, const std::vector<float>& x,
        const std::array<std::vector<float>, 2>& values) {
    std::array<error, 2> errors{};
    if (o.oracle == "none") return errors;
    const auto rows = selected(o.rows, o.oracle == "full" ? o.rows : o.sample_rows);
    const auto columns = selected(o.columns, o.oracle == "full" ? o.columns : o.sample_columns);
    require(uint64_t(rows.size()) * columns.size() <= o.oracle_max_operations / uint64_t(o.k),
        "Requested F64 oracle exceeds --oracle-max-operations; choose explicit sampling or increase its bound");
    std::vector<double> decoded(size_t(o.k), 0);
    for (int row : rows) {
        for (int k = 0; k < o.k; ++k) decoded[size_t(k)] = weight(w.data() + (size_t(row) * (o.k / 256) + k / 256) * 176, k % 256);
        for (int column : columns) {
            double sum = 0;
            for (int k = 0; k < o.k; ++k) sum += decoded[size_t(k)] * double(x[size_t(column) * o.k + k]);
            for (int variant = 0; variant < 2; ++variant) {
                auto& e = errors[size_t(variant)]; const double d = values[size_t(variant)][size_t(column) * o.rows + row] - sum;
                ++e.count; e.maximum = std::max(e.maximum, std::abs(d)); e.squared += d * d; e.reference_squared += sum * sum;
            }
        }
    }
    return errors;
}
double relative(const error& e) { return e.reference_squared == 0 ? std::sqrt(e.squared) : std::sqrt(e.squared / e.reference_squared); }
template<class T> void binary(const std::filesystem::path& path, const std::vector<T>& values) {
    std::ofstream stream(path, std::ios::binary | std::ios::out | std::ios::trunc);
    require(bool(stream), "Cannot create binary output");
    stream.write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * sizeof(T)));
    require(bool(stream), "Cannot finish binary output");
}
double execute(ggml_backend_t backend, ggml_cgraph* graph, int iterations) {
    ggml_backend_synchronize(backend);
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Backend graph execution failed");
    ggml_backend_synchronize(backend);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / iterations;
}
double percentile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end()); return values[size_t(std::ceil(p * values.size())) - 1];
}
int run(const options& o) {
    backends b;
    if (o.backend == "cuda") {
#ifdef TSG_GGML_USE_CUDA
        if (o.device >= ggml_backend_cuda_get_device_count()) { std::fprintf(stderr, "UNAVAILABLE: requested CUDA device is absent; no benchmark passed\n"); return 77; }
        b.original = ggml_backend_cuda_init(o.device);
        require(b.original != nullptr, "Cannot initialize requested CUDA backend");
        b.owned = tsg_dsv4_fused_backend_init(b.original);
#else
        std::fprintf(stderr, "UNAVAILABLE: CUDA support was not compiled; no benchmark passed\n"); return 77;
#endif
    } else {
        b.original = ggml_backend_cpu_init();
        require(b.original != nullptr, "Cannot initialize CPU backend");
        ggml_backend_cpu_set_n_threads(b.original, o.threads); b.owned = b.original;
    }
    require(b.owned != nullptr, "Cannot create owned borrowed-backend wrapper");
    storage s; ggml_init_params init{};
    init.mem_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false) * 2 + 4096;
    init.no_alloc = true; s.ctx = ggml_init(init); require(s.ctx != nullptr, "Cannot create tensor metadata");
    require(ggml_type_size(GGML_TYPE_Q5_K) == 176 && ggml_blck_size(GGML_TYPE_Q5_K) == 256, "Unsupported Q5_K wire format");
    auto* w = ggml_new_tensor_2d(s.ctx, GGML_TYPE_Q5_K, o.k, o.rows);
    auto* x = ggml_new_tensor_2d(s.ctx, GGML_TYPE_F32, o.k, o.columns);
    auto* owned = tsg_matmul_quant_f32(s.ctx, w, x);
    auto* reference = ggml_mul_mat(s.ctx, w, x);
    std::array<ggml_tensor*, 2> outputs{owned, reference};
    std::array<ggml_cgraph*, 2> graphs{};
    for (int v = 0; v < 2; ++v) { graphs[size_t(v)] = ggml_new_graph_custom(s.ctx, 16, false); ggml_build_forward_expand(graphs[size_t(v)], outputs[size_t(v)]); }
    s.buffer = ggml_backend_alloc_ctx_tensors(s.ctx, b.original); require(s.buffer != nullptr, "Cannot allocate resident tensors");
    const auto weights = make_weights(o); const auto input = make_input(o);
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size()); ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    std::array<ggml_backend_t, 2> executors{b.owned, b.original};
    // First execution and backend laziness/capture preparation are excluded.
    // Both variants are warmed before alternating measured order begins.
    for (int v = 0; v < 2; ++v) execute(executors[size_t(v)], graphs[size_t(v)], o.warmup);
    std::array<std::vector<double>, 2> timings;
    for (int repeat = 0; repeat < o.repeats; ++repeat) for (int order = 0; order < 2; ++order) {
        const int v = (repeat % 2 == 0) ? order : 1 - order;
        timings[size_t(v)].push_back(execute(executors[size_t(v)], graphs[size_t(v)], o.iterations));
    }
    std::array<std::vector<float>, 2> values;
    for (int v = 0; v < 2; ++v) {
        values[size_t(v)].resize(size_t(o.rows) * o.columns);
        ggml_backend_tensor_get(outputs[size_t(v)], values[size_t(v)].data(), 0, values[size_t(v)].size() * sizeof(float));
        require(std::all_of(values[size_t(v)].begin(), values[size_t(v)].end(), [](float f) { return std::isfinite(f); }), "Nonfinite output");
    }
    std::vector<unsigned char> retained_weights(weights.size()); std::vector<float> retained_input(input.size());
    ggml_backend_tensor_get(w, retained_weights.data(), 0, retained_weights.size());
    ggml_backend_tensor_get(x, retained_input.data(), 0, retained_input.size() * sizeof(float));
    require(retained_weights == weights && std::memcmp(retained_input.data(), input.data(), input.size() * sizeof(float)) == 0, "Resident sources changed");
    const auto errors = oracle(o, weights, input, values);
    const bool oracle_passed = o.oracle != "none" && errors[0].maximum <= o.max_abs_gate && relative(errors[0]) <= o.relative_l2_gate;
    require(std::filesystem::create_directories(o.output), "Output directory appeared during execution; refusing to overwrite it");
    binary(std::filesystem::path(o.output) / "owned.f32le.bin", values[0]);
    binary(std::filesystem::path(o.output) / "ordinary.f32le.bin", values[1]);
    binary(std::filesystem::path(o.output) / "weights.q5_k.bin", weights);
    binary(std::filesystem::path(o.output) / "input.f32le.bin", input);
    std::ofstream report(std::filesystem::path(o.output) / "measurement.json"); require(bool(report), "Cannot create measurement report");
    report << std::setprecision(17) << "{\n  \"schema\": \"tensorsharp.quant-projection-bench.v1\",\n"
        << "  \"execution_passed\": true,\n  \"backend\": \"" << o.backend << "\",\n  \"device\": " << o.device
        << ",\n  \"k\": " << o.k << ",\n  \"rows\": " << o.rows << ",\n  \"columns\": " << o.columns
        << ",\n  \"warmup\": " << o.warmup << ",\n  \"repeats\": " << o.repeats << ",\n  \"iterations_per_repeat\": " << o.iterations
        << ",\n  \"resident_tensor_payload_bytes\": " << weights.size() + input.size() * 4 + values[0].size() * 8
        << ",\n  \"backend_tensor_buffer_bytes\": " << ggml_backend_buffer_get_size(s.buffer)
        << ",\n  \"allocation_upload_timed\": false,\n  \"timing\": \"wall_graph_compute_and_backend_synchronize\",\n"
        << "  \"gpu_event_ms\": null,\n  \"gpu_event_unavailable_reason\": \"The benchmark does not access the original backend stream; unrelated-stream events are not used\",\n"
        << "  \"graph_capture_state\": \"backend_controlled_not_asserted\",\n  \"source_bytes_unchanged\": true,\n"
        << "  \"oracle\": \"" << o.oracle << "\",\n  \"oracle_values\": " << errors[0].count
        << ",\n  \"oracle_complete_outputs\": " << (o.oracle == "full" ? "true" : "false")
        << ",\n  \"owned_oracle_passed\": " << (o.oracle == "none" ? "null" : oracle_passed ? "true" : "false")
        << ",\n  \"max_abs_gate\": " << o.max_abs_gate << ",\n  \"relative_l2_gate\": " << o.relative_l2_gate
        << ",\n  \"ordinary_is_numerical_oracle\": false,\n  \"model_quality_performance_hardcap_claim\": false,\n  \"variants\": {\n";
    for (int v = 0; v < 2; ++v) {
        report << "    \"" << (v == 0 ? "owned" : "ordinary_performance_reference") << "\": {\"wall_p50_ms\": " << percentile(timings[size_t(v)], .5)
            << ", \"wall_p95_ms\": " << percentile(timings[size_t(v)], .95)
            << ", \"oracle_max_abs\": " << errors[size_t(v)].maximum << ", \"oracle_relative_l2\": " << relative(errors[size_t(v)])
            << ", \"wall_samples_ms\": [";
        for (size_t i = 0; i < timings[size_t(v)].size(); ++i) report << (i ? ", " : "") << timings[size_t(v)][i];
        report << "]}" << (v == 0 ? "," : "") << "\n";
    }
    report << "  }\n}\n"; report.close(); require(bool(report), "Cannot finish measurement report");
    std::printf("EXECUTED: complete outputs saved; owned F64 oracle=%s values=%zu; ordinary is performance reference only\n", o.oracle == "none" ? "not-run" : oracle_passed ? "pass" : "fail", errors[0].count);
    return o.oracle != "none" && !oracle_passed ? 2 : 0;
}
} // namespace
int main(int argc, char** argv) {
    try { return run(parse(argc, argv)); }
    catch (const std::exception& e) { std::fprintf(stderr, "ERROR: %s\n", e.what()); return 1; }
}
