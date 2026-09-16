// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "dsv41_retention.h"
#include <cstdio>
#include <cstdlib>

static void require(bool result, const char * name)
{
    if (!result) { std::fprintf(stderr, "FAIL %s\n", name); std::exit(1); }
}
int main()
{
    require(dsv41_retention_fits(10, 20, 15, 2, 50, true, 30, 5), "exact inclusive bounds");
    require(!dsv41_retention_fits(10, 20, 15, 2, 49, true, 30, 5), "retained cache budget");
    require(!dsv41_retention_fits(10, 21, 15, 2, 50, true, 30, 5), "graph arena budget");
    require(!dsv41_retention_fits(10, 20, 15, 2, 50, true, 29, 5), "next allocation headroom");
    require(!dsv41_retention_fits(10, 0, 0, 0, 100, true, 9, 0), "next cache exceeds free");
    require(!dsv41_retention_fits(10, 0, 11, 0, 100, true, 20, 0), "graph exceeds remainder");
    require(!dsv41_retention_fits(10, 0, 0, 0, 0, false, 0, 0), "disabled budget");
    require(dsv41_retention_fits(10, 20, 15, 2, 50, false, 0, UINT64_MAX), "CPU explicit byte budget");
    require(!dsv41_retention_fits(UINT64_MAX, 0, 0, 1, UINT64_MAX, false, 0, 0), "multiplication overflow");
    require(!dsv41_retention_fits(0, 0, 0, UINT64_MAX, UINT64_MAX, false, 0, 0), "count overflow");
    require(!dsv41_retention_fits(1, UINT64_MAX, 0, 0, UINT64_MAX, false, 0, 0), "sum overflow");
    require(!dsv41_retention_fits(1, 0, UINT64_MAX, 0, UINT64_MAX, true, UINT64_MAX, 0), "headroom overflow");
    require(dsv41_retention_fits(UINT64_MAX, 0, 0, 0, UINT64_MAX, true, UINT64_MAX, 0), "maximum exact fit");
    std::puts("13 retention budget/overflow checks passed");
}
