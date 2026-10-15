// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Standalone CPU canary. The supervisor must install a cgroup before execution.
#include "../../TensorSharp.GGML.Native/glm_host_mapping.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

int main(int argc, char ** argv) {
    if (argc != 3 || (std::strcmp(argv[1], "observe") && std::strcmp(argv[1], "overrun"))) return 2;
    const uint64_t ceiling = std::strtoull(argv[2], nullptr, 10);
    if (ceiling < (64ull << 20) || ceiling > (256ull << 20)) return 2;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string membership;
    if (!tsg_glm_host::read_text("/proc/self/cgroup", membership)) return 2;
    if (!membership.empty() && membership.back() == '\n') membership.pop_back();
    if (membership.find_first_of("\n\r\"\\") != std::string::npos) return 2;
    const uint64_t before = tsg_glm_host::host_available_bytes();
    std::printf("{\"mode\":\"%s\",\"availableBefore\":%llu,\"knownBefore\":%s,\"parentLimit\":%llu,\"membership\":\"%s\"}\n",
        argv[1], (unsigned long long)before, before ? "true" : "false", (unsigned long long)ceiling, membership.c_str());
    if (before > ceiling || tsg_glm_host::can_prefault(87ull << 30, before, 8ull << 30)) return 3;
    const bool overrun = std::strcmp(argv[1], "overrun") == 0;
    const size_t bytes = static_cast<size_t>(overrun ? ceiling + ceiling / 2 : ceiling / 4);
    try {
        std::unique_ptr<unsigned char[]> memory(new unsigned char[bytes]);
        volatile unsigned char * pages = memory.get();
        for (size_t offset = 0; offset < bytes; offset += 4096) pages[offset] = 37;
        pages[bytes - 1] = 53;
        const uint64_t after = tsg_glm_host::host_available_bytes();
        std::printf("{\"touchedBytes\":%llu,\"availableAfter\":%llu,\"knownAfter\":%s}\n",
            (unsigned long long)bytes, (unsigned long long)after, after ? "true" : "false");
        if (overrun) return 4; // A successful overrun is an enforcement failure.
        if (after > ceiling || (before && after && after >= before)) return 5;
        return 0;
    } catch (const std::bad_alloc &) {
        std::fputs("Allocation refused before a verified cgroup OOM kill.\n", stderr);
        return 6;
    }
}
