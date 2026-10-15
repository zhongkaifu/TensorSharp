// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../ggml_ops_moe_file_pipeline_config.h"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace {
int checks = 0;
void require(bool value) { ++checks; if (!value) std::abort(); }
template<class Action> void fails(Action action) {
    bool caught = false;
    try { action(); } catch (const std::runtime_error&) { caught = true; }
    require(caught);
}
}
int main() {
    using namespace tsg;
    HostMoeFilePipelineConfiguration config;
    require(!config.active() && !config.settings().enabled);
    int releases = 0;
    const auto release = [&] { ++releases; };
    for (auto value : {HostMoeFilePipelineSettings{true, 32, 1, 128},
            {true, (1 << 20) + 1, 1, 128}, {true, 64, 0, 128}, {true, 64, 1, 0},
            {true, 64, 1, 127}, {true, 64, 1, 129}, {true, 64, 1, (16u << 20) + 128}})
        require(config.begin(value, release) == 0);
    require(releases == 0 && !config.active());
    fails([&] { config.begin({true, 64, 32u << 20, 16u << 20}, [] {
        throw std::runtime_error("injected begin drain failure"); }); });
    require(!config.active());
    const auto token = config.begin({true, 64, 32u << 20, 16u << 20}, release);
    require(token != 0 && config.owns(token) && releases == 1);
    require(config.begin({false, 64, 1, 128}, release) == 0 && releases == 1);
    const auto policy = config.settings();
    require(host_moe_file_pipeline_eligible(policy, 64, 32u << 20, true, true, true, true));
    require(!host_moe_file_pipeline_eligible(policy, 63, 32u << 20, true, true, true, true));
    require(!host_moe_file_pipeline_eligible(policy, 64, (32u << 20) - 1, true, true, true, true));
    require(!host_moe_file_pipeline_eligible(policy, 64, 32u << 20, false, true, true, true));
    require(!host_moe_file_pipeline_eligible(policy, 64, 32u << 20, true, false, true, true));
    require(!host_moe_file_pipeline_eligible(policy, 64, 32u << 20, true, true, false, true));
    require(!host_moe_file_pipeline_eligible(policy, 64, 32u << 20, true, true, true, false));
    require(!config.end(token + 1, release) && releases == 1);
    fails([&] { config.end(token, [] { throw std::runtime_error("injected end drain failure"); }); });
    require(config.owns(token) && config.settings().enabled);
    require(config.end(token, release) && releases == 2 && !config.active());
    const auto disabled = config.begin({false, 64, 32u << 20, 16u << 20}, release);
    require(disabled != token && disabled != 0 && !config.settings().enabled);
    require(!host_moe_file_pipeline_eligible(config.settings(), 512, 1u << 30, true, true, true, true));
    require(!config.end(token, release) && config.owns(disabled));
    require(config.end(disabled, release));
    for (std::size_t ring : {std::size_t(128), std::size_t(1024), std::size_t(1u << 20), std::size_t(16u << 20)})
        for (std::size_t payload : {std::size_t(0), std::size_t(127), std::size_t(128), std::size_t(129),
                std::size_t(1023), std::size_t(32u << 20), std::size_t(1u << 30)}) {
            const auto slot = host_moe_file_pipeline_slot_bytes(payload, ring);
            require(slot % 64 == 0 && slot <= (8u << 20) && slot * 2 <= ring && slot * 2 <= payload);
        }
    HostMoeFilePipelineConfiguration exhausted(UINT64_MAX);
    const auto last = exhausted.begin({}, [] {});
    require(last == UINT64_MAX && exhausted.end(last, [] {}));
    require(exhausted.begin({}, [] {}) == 0);
    std::printf("host MoE pipeline configuration: %d checks passed (CPU policy/lifecycle only)\n", checks);
}
