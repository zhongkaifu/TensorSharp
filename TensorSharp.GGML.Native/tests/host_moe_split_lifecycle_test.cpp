// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Reuse the shared cache fixture's resident oracle and failure ledger.
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "host_moe_cache_lifecycle_test_hooks.h"
#define main host_moe_shared_fixture_main
#include "host_moe_expert_cache_test.cpp"
#undef main
namespace {
#include "host_moe_split_lifecycle_fixture.inc"
#include "host_moe_group_quota_fixture.inc"
#include "host_moe_phase_quota_fixture.inc"
}
int main(int argc, char** argv) {
    bool lifecycle = false, groups = false, phases = false, direct_reads = false;
    for (int i = 1; i < argc; ++i) lifecycle |= std::strcmp(argv[i], "--split-lifecycle") == 0;
    for (int i = 1; i < argc; ++i) groups |= std::strcmp(argv[i], "--group-quotas") == 0;
    for (int i = 1; i < argc; ++i) phases |= std::strcmp(argv[i], "--phase-quotas") == 0;
    if (!lifecycle && !groups && !phases) return host_moe_shared_fixture_main(argc, argv);
    try {
        std::array<ggml_type, 3> types{GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0};
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--split-lifecycle") == 0 || std::strcmp(argv[i], "--group-quotas") == 0 || std::strcmp(argv[i], "--phase-quotas") == 0) continue;
            if (std::strcmp(argv[i], "--direct-reads") == 0) { direct_reads = true; continue; }
            if (std::strcmp(argv[i], "--quantization") == 0 && i + 1 < argc) {
                ++i; ggml_type type = GGML_TYPE_COUNT;
                if (std::strcmp(argv[i], "iq1_m") == 0) type = GGML_TYPE_IQ1_M;
                if (std::strcmp(argv[i], "iq2_xxs") == 0) type = GGML_TYPE_IQ2_XXS;
                require(type != GGML_TYPE_COUNT, "Unknown split lifecycle quantization");
                types = {type, type, GGML_TYPE_IQ4_NL};
            } else throw std::runtime_error("Unsupported split lifecycle argument");
        }
#ifdef TSG_GGML_USE_CUDA
        set_environment("TS_HOST_MOE_EXPERT_CACHE_MB", "32");
        set_environment("TS_HOST_MOE_EXPERT_CACHE_LAYERS", "1");
        set_environment("TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS", "1");
        set_environment("TS_HOST_MOE_EXPERT_CACHE_PREFETCH", "0");
        set_environment("TS_HOST_MOE_EXPERT_CACHE_LFU", "0");
        set_environment("TS_HOST_MOE_EXPERT_CACHE_PREFILL_ROWS", "32");
        require(!direct_reads || phases, "Direct phase proof requires complete phase quotas fixture");
        set_environment("TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS", direct_reads ? "1" : "0");
        if (ggml_backend_cuda_get_device_count() < 1) {
            std::fprintf(stderr, "SKIP: split lifecycle requires an NVIDIA CUDA device\n"); return 77;
        }
        g_backend = ggml_backend_cuda_init(0);
        require(g_backend != nullptr, "Cannot initialize split lifecycle CUDA backend");
        if (phases) { glm_phase_quota_checks(false, direct_reads); glm_phase_quota_checks(true, direct_reads); }
        else if (groups) glm_group_quota_checks();
        else for (float clamp : {0.0f, 0.0001f}) glm_split_owner_lifecycle_checks(types, clamp);
        ggml_backend_free(g_backend); g_backend = nullptr;
        if (direct_reads && !tsg::g_test_direct_file_reads) return 77;
        return 0;
#else
        std::fprintf(stderr, "SKIP: split lifecycle target was built without CUDA\n"); return 77;
#endif
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        tsg::host_moe_expert_cache_release();
        if (g_backend) { ggml_backend_free(g_backend); g_backend = nullptr; }
        return 1;
    }
}
