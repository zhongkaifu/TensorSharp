// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../ggml_ops_host_moe_cache_policy.h"
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

int main()
{
    int checks = 0;
    const auto require = [&](bool condition) {
        ++checks;
        if (!condition) { std::fprintf(stderr, "FAIL bounded host MoE policy check %d\n", checks); std::exit(1); }
    };
    const char* invalid[] = {nullptr, "", "0", "1", "8", "129", "256", "512", "+128", " 128", "128 ", "-1",
                            "18446744073709551616", "128x"};
    for (const char* value : invalid)
        require(tsg::host_moe_cache_prefill_rows(value) == 8);
    for (int rows : {9, 31, 63, 64, 65, 88, 127, 128})
        require(tsg::host_moe_cache_prefill_rows(std::to_string(rows).c_str()) == rows);
    require(tsg::kHostMoeCacheMaxRouteElements * sizeof(float) == 16384);
    require(tsg::kHostMoeCacheMaxGlmRouteElements * sizeof(float) == 32768);
    for (int rows = -1; rows <= 1025; ++rows) for (bool bulk : {false, true}) for (bool independent : {false, true})
        require(tsg::host_moe_cache_direct_read_phase(bulk, rows, independent)
            == (bulk && !independent && rows > 8 && rows <= 1024));
    require(tsg::host_moe_cache_routes_fit(88, 10));
    require(tsg::host_moe_cache_routes_fit(128, 10));
    require(tsg::host_moe_cache_routes_fit(64, 64));
    require(tsg::host_moe_cache_routes_fit(128, 32));
    require(!tsg::host_moe_cache_routes_fit(65, 64));
    require(!tsg::host_moe_cache_routes_fit(128, 33));
    for (int invalid : {0, -1, (std::numeric_limits<int>::min)(), (std::numeric_limits<int>::max)()})
    {
        require(!tsg::host_moe_cache_routes_fit(invalid, 64));
        require(!tsg::host_moe_cache_routes_fit(64, invalid));
    }
    // Exercise every supported row/selected pair, including the exact boundary.
    for (int rows = 1; rows <= 129; ++rows) for (int selected = 1; selected <= 64; ++selected)
        require(tsg::host_moe_cache_routes_fit(rows, selected) == (rows * selected <= 4096));
    for (int rows = 1; rows <= 1025; ++rows) for (int selected = 1; selected <= 64; ++selected) {
        require(tsg::host_moe_cache_prefill_fits(rows, selected, true) == (rows <= 1024 && rows * selected <= 8192));
        require(tsg::host_moe_cache_prefill_fits(rows, selected, false) == (rows <= 128 && rows * selected <= 4096));
    }
    require(tsg::host_moe_cache_prefill_fits(1024, 8, true));
    require(!tsg::host_moe_cache_prefill_fits(1025, 8, true));
    require(!tsg::host_moe_cache_prefill_fits(1024, 9, true));
    require(tsg::host_moe_cache_prefill_fits(128, 64, true));
    require(!tsg::host_moe_cache_prefill_fits(129, 64, true));
    require(!tsg::host_moe_cache_prefill_fits(128, 64, false));
    const std::int32_t disjoint[]{0, 1, 1, 2, 2, 3};
    const std::int32_t repeated[]{0, 1, 0, 1, 0, 1};
    require(tsg::host_moe_cache_prefill_group_rows(disjoint, 2, 3, 3, 4) == 3);
    require(tsg::host_moe_cache_prefill_group_rows(disjoint, 2, 3, 3, 3) == 2);
    require(tsg::host_moe_cache_prefill_group_rows(disjoint, 2, 3, 3, 2) == 1);
    require(tsg::host_moe_cache_prefill_group_rows(disjoint, 2, 1, 3, 4) == 1);
    require(tsg::host_moe_cache_prefill_group_rows(repeated, 2, 3, 3, 2) == 3);
    require(tsg::host_moe_cache_prefill_group_rows(repeated, 2, 3, 3, 1) == 0);
    require(tsg::host_moe_cache_prefill_group_rows(nullptr, 2, 3, 3, 4) == 0);
    require(tsg::host_moe_cache_prefill_group_rows(disjoint, 0, 3, 3, 4) == 0);
    require(tsg::host_moe_cache_prefill_group_rows(disjoint, 65, 3, 3, 4) == 0);
    require(tsg::host_moe_cache_prefill_group_rows(disjoint, 2, 0, 3, 4) == 0);
    require(tsg::host_moe_cache_prefill_group_rows(disjoint, 2, 3, 0, 4) == 0);
    require(tsg::host_moe_cache_prefill_group_rows(disjoint, 2, 3, 9, 4) == 0);
    std::printf("PASS %d host MoE row/route policy checks; default8, Qwen128/16KiB, explicit GLM1024/32KiB\n", checks);
}
