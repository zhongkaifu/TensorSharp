// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_qwen4exp_attn_diagnostic.h"
#include "ggml-cpu.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <type_traits>

static int checks = 0;
static void require(bool value) { ++checks; if (!value) throw std::runtime_error("attention diagnostic invariant failed"); }
template<class F> static void rejected(F action) {
    bool failed = false; try { action(); } catch (const std::invalid_argument&) { failed = true; }
    require(failed);
}
static void environment(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
int main() {
    namespace fs = std::filesystem;
    const auto directory = fs::temp_directory_path() / ("q4e-attn-diagnostic-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(directory);
    std::vector<fs::path> files;
    try {
        require(!tsg::qwen4exp_attn_diagnostic_parse(nullptr, nullptr, nullptr).enabled());
        for (const char* invalid : {"", "-1", "+1", "1x", " 1", "4096", "999999999999999999999"})
            rejected([&] { tsg::qwen4exp_attn_diagnostic_parse(invalid, "77", "out"); });
        rejected([] { tsg::qwen4exp_attn_diagnostic_parse("39", nullptr, "out"); });
        rejected([] { tsg::qwen4exp_attn_diagnostic_parse("39", "2147483648", "out"); });
        rejected([] { tsg::qwen4exp_attn_diagnostic_parse("39", "77", ""); });
        const auto parsed = tsg::qwen4exp_attn_diagnostic_parse("39", "77", "out");
        require(parsed.layer == 39 && parsed.position == 77);
        require(parsed.matches(false, 39) && parsed.matches(true, 39));
        require(!parsed.matches(false, 38) && !parsed.matches(true, 38));
        for (const auto* scope : {"both", "solo", "arena"}) {
            const auto selected = tsg::qwen4exp_attn_diagnostic_parse("39", "77", "out", scope);
            require(selected.matches(false, 39) == (std::strcmp(scope, "arena") != 0));
            require(selected.matches(true, 39) == (std::strcmp(scope, "solo") != 0));
        }
        for (const auto* scope : {"", "SOLO", "arenax", " both"})
            rejected([&] { tsg::qwen4exp_attn_diagnostic_parse("39", "77", "out", scope); });
        rejected([] { tsg::qwen4exp_attn_diagnostic_parse(nullptr, nullptr, nullptr, "arena"); });
        const int64_t valid[4] = {256, 1502, 2, 1};
        require(tsg::qwen4exp_attn_diagnostic_bytes(valid, 2) == 1538048);
        for (const int64_t bad : {INT64_C(0), INT64_C(-1), INT64_MAX}) {
            const int64_t shape[4] = {256, bad, INT64_MAX, 1};
            bool failed = false;
            try { (void)tsg::qwen4exp_attn_diagnostic_bytes(shape, 2); } catch (const std::runtime_error&) { failed = true; }
            require(failed);
        }
        environment("TS_Q4E_ATTN_TRACE_LAYER", "39"); environment("TS_Q4E_ATTN_TRACE_POSITION", "77");
        environment("TS_Q4E_ATTN_TRACE_DIR", directory.string().c_str());
        environment("TS_Q4E_ATTN_TRACE_SCOPE", "arena");
        require(!tsg::qwen4exp_attn_diagnostic_config().matches(false, 39));
        require(tsg::qwen4exp_attn_diagnostic_config().matches(true, 39));
        std::unique_ptr<std::remove_pointer_t<ggml_backend_t>, decltype(&ggml_backend_free)>
            backend(ggml_backend_cpu_init(), ggml_backend_free);
        ggml_init_params params{ggml_tensor_overhead() * 16, nullptr, true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
        require(backend && ctx);
        auto* source = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 4, 6, 2);
        auto* view = ggml_view_3d(ctx.get(), source, 4, 3, 2, source->nb[1], source->nb[2], 4 * sizeof(float));
        tsg::Qwen4ExpAttnDiagnostic diagnostic; diagnostic.layer = 39;
        diagnostic.add("unprotected", view, false);
        require(diagnostic.items.size() == 1 && !diagnostic.items[0].readable);
        require((source->flags & GGML_TENSOR_FLAG_OUTPUT) == 0);
        require((view->flags & GGML_TENSOR_FLAG_OUTPUT) == 0);
        diagnostic.items.clear(); diagnostic.add("strided", view, true);
        diagnostic.add("mask", source, false);
        require(!diagnostic.items.back().readable);
        require((source->flags & GGML_TENSOR_FLAG_OUTPUT) == 0);
        std::unique_ptr<std::remove_pointer_t<ggml_backend_buffer_t>, decltype(&ggml_backend_buffer_free)>
            buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
        require(buffer != nullptr);
        std::vector<float> values(48); for (int i = 0; i < 48; ++i) values[i] = static_cast<float>(i);
        ggml_backend_tensor_set(source, values.data(), 0, values.size() * sizeof(float));
        diagnostic.dump("solo", 77, 0);
        require(fs::is_empty(directory)); // excluded scope performs no writes/downloads.
        diagnostic.dump("arena", 76, 0);
        require(fs::is_empty(directory)); // the position gate performs no writes/downloads.
        diagnostic.dump("arena", 77, 0);
        for (const auto& entry : fs::directory_iterator(directory)) files.push_back(entry.path());
        require(files.size() == 1);
        std::ifstream input(files[0], std::ios::binary);
        const std::vector<char> packed((std::istreambuf_iterator<char>(input)), {});
        require(packed.size() == 24 * sizeof(float));
        for (int i = 0; i < 24; ++i) {
            float actual; std::memcpy(&actual, packed.data() + i * sizeof(float), sizeof(float));
            require(actual == static_cast<float>(i < 12 ? i + 4 : i + 16));
        }
        input.close();
        for (const auto& file : files) fs::remove(file);
        fs::remove(directory);
        std::printf("PASS %d checks; CPU-only diagnostic parsing/no-pin/strided readback\n", checks);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s; evidence directory=%s\n", e.what(), directory.string().c_str());
        return 1;
    }
}
