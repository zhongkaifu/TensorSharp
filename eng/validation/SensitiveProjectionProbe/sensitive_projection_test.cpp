// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_precision_policy.h"
#include "ggml_ops_matmul_precision.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#include "ggml_ops_dsv4_fused.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <random>
#include <vector>
#include "precision_test_utils.h"

#ifdef TSG_GGML_USE_CUDA
// This allocator replacement belongs only to the standalone validation binary.
// Faults/tracking are enabled solely around the main thread's constructor calls;
// ordinary oracle execution and CUDA/ggml worker threads remain unarmed. Fixed
// storage avoids allocations or recursive hooks inside the tracker itself.
namespace constructor_fault {
constexpr size_t maximum_pointers = 32;
struct state {
    bool armed = false, overflow = false;
    size_t fail_on = 0, attempts = 0, live = 0;
    void* pointers[maximum_pointers] = {};
};
thread_local state current;

void* allocate(size_t size) {
    auto& test = current;
    if (test.armed && ++test.attempts == test.fail_on) throw std::bad_alloc();
    void* pointer;
    while (!(pointer = std::malloc(size ? size : 1))) {
        const auto handler = std::get_new_handler();
        if (!handler) throw std::bad_alloc();
        handler();
    }
    if (test.armed) {
        if (test.live == maximum_pointers) {
            test.overflow = true;
            std::free(pointer);
            throw std::bad_alloc();
        }
        test.pointers[test.live++] = pointer;
    }
    return pointer;
}
void release(void* pointer) noexcept {
    auto& test = current;
    for (size_t i = 0; i < test.live; ++i)
        if (test.pointers[i] == pointer) {
            test.pointers[i] = test.pointers[--test.live];
            test.pointers[test.live] = nullptr;
            break;
        }
    std::free(pointer);
}
}
void* operator new(size_t size) { return constructor_fault::allocate(size); }
void* operator new[](size_t size) { return constructor_fault::allocate(size); }
void operator delete(void* pointer) noexcept { constructor_fault::release(pointer); }
void operator delete[](void* pointer) noexcept { constructor_fault::release(pointer); }
void operator delete(void* pointer, size_t) noexcept { constructor_fault::release(pointer); }
void operator delete[](void* pointer, size_t) noexcept { constructor_fault::release(pointer); }
#endif

// CPU uses the unchanged upstream backend. CUDA uses the existing owned
// executor's private pedantic handle; GGML_PREC_F32 on ordinary CUDA still
// permits TF32. Compare complete gate projections to independent FP64 math.
static void check_environment()
{
    for (const char* value : {"f16", "fp16", "bf16", "F16", "Fp16", "BF16"})
        require(tsg_f32_projection_environment_conflicts(value), "Conflicting cuBLAS override accepted");
    for (const char* value : std::array<const char*, 7>{nullptr, "", "auto", "f32", "fp32", "F32", "unexpected"})
        require(!tsg_f32_projection_environment_conflicts(value), "Non-conflicting cuBLAS setting refused");
}

struct result { std::vector<float> first, second; };

static result check_chain(ggml_backend_t backend, int tokens, bool precise, bool owned_cuda)
{
    constexpr int inner = 256, rank = 64, rows = 128;
    auto* ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "Cannot construct sensitive projection context");
    input_tensor a(ctx, GGML_TYPE_BF16, {inner, rank, 1, 1});
    input_tensor b(ctx, GGML_TYPE_BF16, {rank, rows, 1, 1});
    input_tensor x(ctx, GGML_TYPE_F32, {inner, tokens, 1, 1});
    const auto role = precise ? tsg_projection_role::recurrent_decay : tsg_projection_role::throughput;
    auto* first_weight = tsg_projection_weight(ctx, a.tensor, role);
    auto* second_weight = tsg_projection_weight(ctx, b.tensor, role);
    require(first_weight->type == (precise ? GGML_TYPE_F32 : GGML_TYPE_BF16)
        && second_weight->type == (precise ? GGML_TYPE_F32 : GGML_TYPE_BF16), "Sensitive role did not widen floating-point weights");
    auto* first = ggml_mul_mat(ctx, first_weight, x.tensor);
    auto* second = ggml_mul_mat(ctx, second_weight, first);
    // Layout-only wrappers must preserve the actual matmul's precision tag.
    tsg_require_projection_precision(ctx, ggml_reshape_2d(ctx, first, rank, tokens), role, owned_cuda);
    tsg_require_projection_precision(ctx, second, role, owned_cuda);
    for (auto* node : {first, second}) {
        if (precise && owned_cuda) require(node->op == GGML_OP_CUSTOM, "CUDA sensitive projection bypassed the strict owned executor");
        else require(node->op == GGML_OP_MUL_MAT
            && ggml_get_op_params_i32(node, 0) == (precise ? GGML_PREC_F32 : GGML_PREC_DEFAULT),
            "Sensitive/throughput role did not preserve its declared precision contract");
    }
    auto* quantized = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, inner, rank);
    require(tsg_projection_weight(ctx, quantized, tsg_projection_role::recurrent_update) == quantized,
        "Sensitive role widened quantized checkpoint weights");
    auto* untouched = ggml_mul_mat(ctx, quantized, x.tensor);
    tsg_require_projection_precision(ctx, untouched, tsg_projection_role::recurrent_update, owned_cuda);
    require(ggml_get_op_params_i32(untouched, 0) == GGML_PREC_DEFAULT, "Sensitive roles changed quantized arithmetic");
    ggml_set_output(first);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, second);
    auto* buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "Cannot allocate sensitive projection fixture");
    std::mt19937 random(0x19a318u);
    std::uniform_real_distribution<float> weights(-0.08f, 0.08f);
    for (float& value : a.values) value = weights(random);
    for (float& value : b.values) value = weights(random);
    for (int column = 0; column < tokens; ++column)
        for (int k = 0; k < inner; ++k)
            x.values[size_t(column) * inner + k] = std::sin(float(k * 13 + column * 7) * 0.019f)
                + float((k + column) % 17) * 0.00013f;
    a.upload(); b.upload(); x.upload();
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Sensitive projection graph failed");
    result actual{std::vector<float>(size_t(rank) * tokens), std::vector<float>(size_t(rows) * tokens)};
    ggml_backend_tensor_get(first, actual.first.data(), 0, actual.first.size() * sizeof(float));
    ggml_backend_tensor_get(second, actual.second.data(), 0, actual.second.size() * sizeof(float));
    std::vector<double> expected_first(actual.first.size()), expected_second(actual.second.size());
    for (int column = 0; column < tokens; ++column) {
        for (int row = 0; row < rank; ++row)
            for (int k = 0; k < inner; ++k)
                expected_first[size_t(column) * rank + row] += double(a.at(k, row)) * x.at(k, column);
        for (int row = 0; row < rows; ++row)
            for (int k = 0; k < rank; ++k)
                expected_second[size_t(column) * rows + row] += double(b.at(k, row)) * expected_first[size_t(column) * rank + k];
    }
    const auto measure = [precise](const std::vector<float>& values, const std::vector<double>& oracle) {
        double max_abs = 0, squared = 0, reference_squared = 0;
        for (size_t i = 0; i < values.size(); ++i) {
            require(std::isfinite(values[i]), "Sensitive projection emitted a nonfinite value");
            const double delta = double(values[i]) - oracle[i];
            max_abs = std::max(max_abs, std::abs(delta)); squared += delta * delta;
            reference_squared += oracle[i] * oracle[i];
        }
        const double relative = std::sqrt(squared / reference_squared);
        // Declared before measurements: F32 SGEMM may change last-bit reduction
        // across shapes, but may not narrow its activation chain to BF16.
        if (precise) require(max_abs <= 5e-5 && relative <= 5e-6, "Sensitive BF16 gate chain lost F32 activation accuracy");
        return std::pair<double, double>{max_abs, relative};
    };
    const auto first_error = measure(actual.first, expected_first), second_error = measure(actual.second, expected_second);
    a.check_unchanged(); b.check_unchanged(); x.check_unchanged();
    std::printf("SENSITIVE_PROJECTION backend=%s rows=%d role=%s full_values=%zu first_max_abs=%.9g first_relative_l2=%.9g second_max_abs=%.9g second_relative_l2=%.9g\n",
        ggml_backend_name(backend), tokens, precise ? "f32-gate" : "throughput-observation",
        actual.first.size() + actual.second.size(), first_error.first, first_error.second, second_error.first, second_error.second);
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
    return actual;
}

#ifdef TSG_GGML_USE_CUDA
static void check_constructor_backend_health(ggml_backend_t backend)
{
    auto* ctx = ggml_init({1024 * 1024, nullptr, true});
    require(ctx != nullptr, "Cannot construct backend health graph");
    input_tensor a(ctx, GGML_TYPE_F32, {32, 1, 1, 1});
    input_tensor b(ctx, GGML_TYPE_F32, {32, 1, 1, 1});
    for (size_t i = 0; i < a.values.size(); ++i) {
        a.values[i] = float(i);
        b.values[i] = float(31 - i);
    }
    auto* sum = ggml_add(ctx, a.tensor, b.tensor);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, sum);
    auto* buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "Backend health allocation failed after constructor fault");
    a.upload(); b.upload();
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
        "Backend health compute failed after constructor fault");
    float actual[32];
    ggml_backend_tensor_get(sum, actual, 0, sizeof(actual));
    for (float value : actual) require(value == 31.0f, "Constructor fault corrupted borrowed CUDA execution");
    a.check_unchanged(); b.check_unchanged();
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void check_constructor_failures(ggml_backend_t original)
{
    // Initialize lazy CUDA/cuBLAS state before selecting the four owned new
    // sites: context, matmul state, device and backend. No GPU fault is injected.
    auto* warmup = tsg_dsv4_fused_backend_init(original);
    require(warmup != nullptr, "Cannot warm up constructor failure fixture");
    check_constructor_backend_health(warmup);
    ggml_backend_free(warmup);
    check_constructor_backend_health(original);
    // Bind fault indices to the completed constructor rather than assuming a
    // vendor's lazy initialization made no additional tracked C++ allocations.
    auto& calibration = constructor_fault::current;
    require(!calibration.armed && calibration.live == 0, "Constructor calibration began with live owners");
    calibration = {};
    calibration.armed = true;
    auto* complete = tsg_dsv4_fused_backend_init(original);
    calibration.armed = false;
    require(complete != nullptr && calibration.attempts == 4 && calibration.live == 4 && !calibration.overflow,
        "Constructor no longer has exactly four bounded owned C++ allocations");
    ggml_backend_free(complete);
    require(calibration.live == 0, "Complete constructor teardown leaked tracked owners");
    for (size_t fail_on = 1; fail_on <= 4; ++fail_on) {
        auto& test = constructor_fault::current;
        require(!test.armed && test.live == 0, "Constructor fixture began with live tracked allocations");
        test = {};
        test.fail_on = fail_on;
        test.armed = true;
        ggml_backend_t unexpected = nullptr;
        bool caught = false;
        try { unexpected = tsg_dsv4_fused_backend_init(original); }
        catch (const std::bad_alloc&) { caught = true; }
        catch (...) {
            test.armed = false;
            require(false, "Constructor fault raised an unexpected exception");
        }
        test.armed = false;
        if (unexpected) ggml_backend_free(unexpected);
        require(caught, "Owned constructor allocation did not raise bad_alloc");
        require(test.attempts == fail_on && !test.overflow, "Constructor failure missed its exact bounded allocation site");
        require(test.live == 0, "Failed constructor leaked an owned C++ allocation");
        // The CUDA backend is borrowed: a failed wrapper must neither free it
        // nor leave its stream unusable for a later complete wrapper.
        auto* recovered = tsg_dsv4_fused_backend_init(original);
        require(recovered != nullptr, "Cannot recreate wrapper after constructor fault");
        check_constructor_backend_health(recovered);
        ggml_backend_free(recovered);
        check_constructor_backend_health(original);
        require(test.live == 0, "Recovery changed constructor allocation ownership");
        std::printf("CONSTRUCTOR_FAULT allocation=%zu caught_bad_alloc=1 tracked_live=0 borrowed_cuda_healthy=1\n", fail_on);
    }
}
#endif

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const bool constructor_failures = argc == 2 && std::strcmp(argv[1], "--constructor-faults") == 0;
    require(argc == 1 || constructor_failures, "Unknown sensitive projection fixture argument");
    check_environment();
#ifdef TSG_GGML_USE_CUDA
    if (ggml_backend_cuda_get_device_count() == 0) return 77;
    require(!tsg_f32_projection_environment_conflicts(std::getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE")),
        "Clear conflicting cuBLAS override before numerical qualification");
    auto* original = ggml_backend_cuda_init(0);
    require(original != nullptr, "Cannot initialize original CUDA backend");
    if (constructor_failures) {
        check_constructor_failures(original);
        ggml_backend_free(original);
        std::printf("PASS: four bounded main-thread C++ constructor failures released all tracked owners; original CUDA backend and recovered wrappers computed and released successfully; no vendor-workspace or fatal-CUDA recovery claim\n");
        return 0;
    }
    auto* backend = tsg_dsv4_fused_backend_init(original);
    constexpr bool owned_cuda = true;
#else
    if (constructor_failures) {
        std::fprintf(stderr, "UNAVAILABLE: constructor failure fixture requires the CUDA executor\n");
        return 77;
    }
    auto* backend = ggml_backend_cpu_init();
    if (backend) ggml_backend_cpu_set_n_threads(backend, 4);
    constexpr bool owned_cuda = false;
#endif
    require(backend != nullptr, "Cannot initialize sensitive projection backend");
    const auto narrow = check_chain(backend, 512, true, owned_cuda), wide = check_chain(backend, 809, true, owned_cuda);
    size_t overlap_values = 0;
    const auto overlap = [&](const auto& a, const auto& b) {
        for (size_t i = 0; i < a.size(); ++i) {
            require(std::abs(double(a[i]) - b[i]) <= 5e-5, "512/809 shared columns changed sensitive gate accuracy");
            ++overlap_values;
        }
    };
    overlap(narrow.first, wide.first); overlap(narrow.second, wide.second);
    check_chain(backend, 512, false, owned_cuda); check_chain(backend, 809, false, owned_cuda);
    ggml_backend_free(backend);
#ifdef TSG_GGML_USE_CUDA
    ggml_backend_free(original);
#endif
    std::printf("PASS: sensitive floating-point projection roles, two-stage BF16/F32 gate chain, complete512/809 outputs and %zu overlap values; FP64 oracle, input/lifecycle and environment-conflict checks; no whole-model quality or hard-cap claim\n", overlap_values);
    return 0;
}
