// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_internal.h"
#include "ggml_ops_bonsai.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

// Compile the production transform source directly so Windows exercises it
// without relying on C++ DLL exports. Backend/error ownership is isolated here;
// the unrelated weight projection must never be reached by a transform test.
namespace tsg {
DeviceState g_device_states[TSG_MAX_DEVICES];
thread_local int g_active_rank = 0;
int g_backend_type = BACKEND_TYPE_CPU;
void set_last_error(const std::string &) {}
ggml_tensor * weight_mul_mat(ggml_context *, ggml_tensor *, ggml_tensor *, const void *) {
    throw std::logic_error("Hadamard-only fixture unexpectedly requested a weight projection");
}
}
extern "C" int TSGgml_BonsaiRegisterWeight(const void *, int, const float *, int, int, int, int, int);
extern "C" void TSGgml_BonsaiUnregisterWeight(const void *);

namespace {
void require(bool condition, const char * message) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

struct registration {
    int key = 0;
    std::vector<float> signs;
    registration(int width, bool inverse, int hd = 0, int nk = 0, int rep = 1) : signs(width) {
        for (int i = 0; i < width; ++i) signs[i] = ((i * 53 + i / 7) % 11) < 5 ? -1.0f : 1.0f;
        require(TSGgml_BonsaiRegisterWeight(&key, width, signs.data(), 1024, inverse ? 1 : 0, hd, nk, rep) == 1,
                "register transform");
    }
    ~registration() { TSGgml_BonsaiUnregisterWeight(&key); }
};

// Dense Sylvester matrix definition, independent of butterfly implementations.
std::vector<float> oracle(const std::vector<float> & input, int width, const registration & t,
                          bool inverse, int hd, int nk, int rep) {
    std::vector<float> expected(input.size());
    for (size_t row = 0; row < input.size() / width; ++row)
        for (int block = 0; block < width; block += 1024)
            for (int out = 0; out < 1024; ++out) {
                double sum = 0;
                for (int in = 0; in < 1024; ++in) {
                    int source = block + in;
                    if (rep > 1) {
                        const int component = source % hd;
                        const int vhead = source / hd;
                        source = component + hd * (vhead / rep + nk * (vhead % rep));
                    }
                    unsigned bits = unsigned(out & in), parity = 0;
                    while (bits) { parity ^= bits & 1; bits >>= 1; }
                    const double value = input[row * width + source] * (inverse ? 1.0f : t.signs[block + in]);
                    sum += parity ? -value : value;
                }
                expected[row * width + block + out] = float(sum / 32.0) * (inverse ? t.signs[block + out] : 1.0f);
            }
    return expected;
}

double compare(const std::vector<float> & actual, const std::vector<float> & expected) {
    require(actual.size() == expected.size(), "output size");
    double error = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double delta = std::abs(double(actual[i]) - expected[i]);
        require(std::isfinite(delta), "nonfinite transform output");
        error = std::max(error, delta);
    }
    return error;
}

void transform_test(ggml_backend_t backend, int width, int rows, bool inverse,
                    bool grouped = false, bool strided = false, bool half = false) {
    const int hd = grouped ? 128 : 0, nk = grouped ? 16 : 0, rep = grouped ? 3 : 1;
    registration t(width, inverse, hd, nk, rep);
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "context");
    const int storage_width = width + (strided ? 16 : 0);
    const ggml_type dtype = half ? GGML_TYPE_F16 : GGML_TYPE_F32;
    auto * storage = ggml_new_tensor_2d(ctx, dtype, storage_width, rows);
    auto * input = strided ? ggml_view_2d(ctx, storage, width, rows, storage->nb[1], 0) : storage;
    std::vector<float> values(size_t(width) * rows), padded(size_t(storage_width) * rows, -999.0f);
    for (int row = 0; row < rows; ++row)
        for (int j = 0; j < width; ++j) {
            float value = float(((j * 53 + row * 71 + j / 1024 * 11) % 4093) - 2046) / 1031.0f;
            if (half) value = ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
            padded[size_t(row) * storage_width + j] = values[size_t(row) * width + j] = value;
        }
    std::vector<float> expected = oracle(values, width, t, inverse, hd, nk, rep);
    tsg::BonsaiGraphScope scope;
    auto * output = tsg::bonsai_transform(ctx, input, &t.key, inverse);
    require(output != input, "registered transform ignored");
    require(tsg::bonsai_transform(ctx, input, &t.key, inverse) == output, "scope failed to share activation");
    int alias_key = 0;
    require(TSGgml_BonsaiRegisterWeight(&alias_key, width, t.signs.data(), 1024, inverse, hd, nk, rep) == 1,
            "register shared transform");
    require(tsg::bonsai_transform(ctx, input, &alias_key, inverse) == output, "equivalent weights did not share activation");
    TSGgml_BonsaiUnregisterWeight(&alias_key);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "backend allocation");
    if (half) {
        std::vector<ggml_fp16_t> encoded(padded.size());
        ggml_fp32_to_fp16_row(padded.data(), encoded.data(), int64_t(encoded.size()));
        ggml_backend_tensor_set(storage, encoded.data(), 0, encoded.size() * sizeof(ggml_fp16_t));
    } else ggml_backend_tensor_set(storage, padded.data(), 0, padded.size() * sizeof(float));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "transform graph compute");
    std::vector<float> actual(values.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    const double error = compare(actual, expected);
    std::printf("%s width=%d rows=%d inverse=%d grouped=%d strided=%d half=%d max_abs_error=%.9g\n",
                ggml_backend_name(backend), width, rows, inverse, grouped, strided, half, error);
    require(error < 2e-5, "Hadamard differs from independent dense oracle");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void scope_test() {
    registration t(5120, false);
    auto * leaves = ggml_init({65536, nullptr, true});
    auto * input = ggml_new_tensor_2d(leaves, GGML_TYPE_F32, 5120, 1);
    auto * first = ggml_init({65536, nullptr, true});
    auto * second = ggml_init({65536, nullptr, true});
    {
        tsg::BonsaiGraphScope outer;
        auto * a = tsg::bonsai_transform(first, input, &t.key, false);
        auto * b = tsg::bonsai_transform(second, input, &t.key, false);
        require(a != b, "memo reused a tensor across distinct graph contexts");
        {
            tsg::BonsaiGraphScope inner;
            require(tsg::bonsai_transform(first, input, &t.key, false) != a, "nested scope did not isolate graph memo");
        }
        require(tsg::bonsai_transform(first, input, &t.key, false) == a, "nested scope did not restore outer memo");
    }
    ggml_free(first);
    ggml_free(second);
    ggml_free(leaves);
    // Repeat after destruction, often reusing the same ggml tensor addresses.
    for (int repeat = 0; repeat < 4; ++repeat) {
        auto * ctx = ggml_init({65536, nullptr, true});
        auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, 1);
        { tsg::BonsaiGraphScope scope; require(tsg::bonsai_transform(ctx, x, &t.key, false)->src[0] != nullptr, "new context transform"); }
        ggml_free(ctx);
    }
}

void validation_test() {
    int key = 0;
    std::vector<float> signs(5120, 1);
    require(TSGgml_BonsaiRegisterWeight(nullptr, 5120, signs.data(), 1024, 0, 0, 0, 1) == 0, "null key accepted");
    require(TSGgml_BonsaiRegisterWeight(&key, 5119, signs.data(), 1024, 0, 0, 0, 1) == 0, "unaligned width accepted");
    require(TSGgml_BonsaiRegisterWeight(&key, 5120, signs.data(), 1024, 1, 128, 16, 3) == 0, "inverse grouping accepted");
    signs[4000] = 0;
    require(TSGgml_BonsaiRegisterWeight(&key, 5120, signs.data(), 1024, 0, 0, 0, 1) == 0, "invalid signs accepted");
    signs[4000] = 1;
    require(TSGgml_BonsaiRegisterWeight(&key, 5120, signs.data(), 1024, 0, 0, 0, 1) == 1, "valid registration failed");
    require(TSGgml_BonsaiRegisterWeight(&key, 5120, signs.data(), 1024, 0, 0, 0, 1) == 0, "duplicate registration accepted");
    auto * ctx = ggml_init({65536, nullptr, true});
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, 1);
    bool rejected = false;
    try { tsg::bonsai_transform(ctx, x, &key, true); } catch (const std::exception &) { rejected = true; }
    require(rejected, "direction mismatch accepted");
    TSGgml_BonsaiUnregisterWeight(&key);
    require(tsg::bonsai_transform(ctx, x, &key, false) == x, "unregistered weight changed activations");
    ggml_free(ctx);
}
} // namespace

int main(int argc, char ** argv) {
    const char * name = argc > 1 ? argv[1] : "cpu";
    ggml_backend_t backend = nullptr;
    if (std::strcmp(name, "cpu") == 0) { backend = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(backend, 4); }
#ifdef TSG_GGML_USE_METAL
    if (std::strcmp(name, "metal") == 0) backend = ggml_backend_metal_init();
#endif
#ifdef TSG_GGML_USE_CUDA
    if (std::strcmp(name, "cuda") == 0) backend = ggml_backend_cuda_init(0);
#endif
    if (!backend) { std::fprintf(stderr, "SKIP: %s backend unavailable\n", name); return 77; }
    tsg::active_backend() = backend;
    tsg::g_backend_type = std::strcmp(name, "metal") == 0 ? tsg::BACKEND_TYPE_METAL
        : std::strcmp(name, "cuda") == 0 ? tsg::BACKEND_TYPE_CUDA : tsg::BACKEND_TYPE_CPU;
    validation_test();
    scope_test();
    for (int width : {5120, 6144, 17408})
        for (int rows : {1, 3})
            for (bool inverse : {false, true}) transform_test(backend, width, rows, inverse);
    transform_test(backend, 6144, 3, false, true);
    transform_test(backend, 5120, 3, false, false, true);
    transform_test(backend, 5120, 3, false, false, true, true);
    transform_test(backend, 5120, 3, true, false, true, true);
    tsg::bonsai_clear_backend();
    ggml_backend_free(backend);
    tsg::active_backend() = nullptr;
    std::puts("PASS: signed Hadamard transforms, grouped heads, noncontiguous inputs, graph scope and registration guards");
    return 0;
}
