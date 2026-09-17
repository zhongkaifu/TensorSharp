// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// DeepSeek V4 / V4.1 load planning without a model: the automatic prefill
// width, the routed-expert offload search it is built on, and the refusal an
// explicit --n-cpu-moe gets. Synthetic budgets pin each rule; the DeepSeek
// V4.1 Flash Q4_K_M sizes with the seven-A40 lane's budgets pin the choice the
// verification session is expected to log.
#include "dsv4_ubatch_plan.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <string>
#include <vector>

using namespace tsg_dsv4_plan;

static int failures = 0;

static void expect(bool condition, const std::string & what)
{
    if (!condition)
    {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

static constexpr size_t GiB = size_t(1) << 30;
static constexpr size_t MiB = size_t(1) << 20;

// ---------------------------------------------------------------------------
// Synthetic model: `layers` layers of 10 GiB (8 GiB of it routed experts), one
// 1 GiB embedding and a 1 GiB head, over `devices` devices. `extra` is added
// to every layer's cache cost and `held` taken from every budget, standing in
// for what a wider ubatch costs.
// ---------------------------------------------------------------------------
static split_costs synthetic(int layers, int devices, size_t free_per_device, size_t held, size_t extra)
{
    split_costs c;
    c.layer_bytes.assign((size_t) layers, 10 * GiB);
    c.layer_exps_bytes.assign((size_t) layers, 8 * GiB);
    c.layer_engram_bytes.assign((size_t) layers, 0);
    c.layer_cache_bytes.assign((size_t) layers, 64 * MiB + extra);
    c.fixed_bytes.assign((size_t) devices, 0);
    c.fixed_bytes[0] += GiB;
    c.fixed_bytes[(size_t) devices - 1] += GiB;
    c.dev_budget.assign((size_t) devices, free_per_device - held);
    return c;
}

static std::vector<ubatch_candidate> candidates_for(const std::vector<split_costs> & costs, int n_cpu_moe_req,
                                                    bool want_engram = false)
{
    std::vector<ubatch_candidate> out;
    const int widths[] = {1024, 512, 256};
    for (size_t i = 0; i < costs.size(); ++i)
    {
        ubatch_candidate c;
        c.ubatch = widths[i];
        c.offload = plan_offload(costs[i], want_engram, n_cpu_moe_req);
        out.push_back(c);
    }
    return out;
}

static void check_candidate_lists()
{
    auto same = [](const std::vector<int> & a, const std::vector<int> & b) { return a == b; };
    expect(same(ubatch_candidates(UBATCH_AUTO, true, 0), {1024, 512, 256}), "auto on accelerators evaluates 1024/512/256");
    expect(same(ubatch_candidates(UBATCH_AUTO, false, 0), {256}), "auto on the CPU device keeps 256");
    for (int explicit_width : {1, 32, 256, 300, 512, 700, 1024, 2048, 4096})
        for (bool accelerators : {false, true})
            expect(same(ubatch_candidates(explicit_width, accelerators, 5), {explicit_width}),
                "an explicit width " + std::to_string(explicit_width) + " is the only candidate");
    expect(same(ubatch_candidates(0, true, 0), {512}), "0 keeps meaning 512");
    expect(same(ubatch_candidates(-7, true, 0), {512}), "other negative widths keep meaning 512");
    // DSpark verify needs block + 1 rows in one micro-batch.
    expect(same(auto_ubatch_candidates(5), {1024, 512, 256}), "DSpark block 5 keeps every candidate");
    for (int block = 1; block <= 2047; ++block)
        for (int width : auto_ubatch_candidates(block))
            expect(width >= block + 1, "DSpark block " + std::to_string(block) + " never gets width " +
                std::to_string(width));
    expect(same(auto_ubatch_candidates(300), {1024, 512}), "a 300-token drafter block drops 256");
    std::printf("CANDIDATES auto={1024,512,256} cpu={256} explicit=unchanged dspark_block5_min=%d\n",
        auto_ubatch_candidates(5).back());
}

static void check_synthetic_choices()
{
    // 8 layers x 10 GiB over two 43 GiB devices with a 2 GiB reserve: the
    // leading two layers' experts go to the host (dev0 takes layers 0..4, dev1
    // 5..7), with ~5.7 GiB of slack before a third layer has to follow.
    // (1) Same offload at every width -> the widest.
    {
        const size_t free_b = 43 * GiB;
        std::vector<split_costs> costs = {
            synthetic(8, 2, free_b, 3 * GiB, 2 * MiB), synthetic(8, 2, free_b, 2 * GiB + 512 * MiB, MiB),
            synthetic(8, 2, free_b, 2 * GiB, 0)};
        auto c = candidates_for(costs, -1);
        auto choice = choose_ubatch(c, 8, -1, false);
        expect(c[0].offload.need_cpu_moe == c[2].offload.need_cpu_moe, "(1) fixture: equal needs");
        expect(choice.index == 0 && !choice.refused, "(1) same offload picks 1024: " + choice.reason);
        std::printf("SAME_OFFLOAD need=%d -> %d (%s)\n", c[2].offload.need_cpu_moe, c[(size_t) choice.index].ubatch,
            choice.reason.c_str());
    }
    // Calibrate a budget where 256 needs N host layers with little slack, so a
    // wider reserve tips over into N + 1.
    const size_t tight = 43 * GiB;
    split_costs probe = synthetic(8, 2, tight, 2 * GiB, 0);
    const int base_need = min_cpu_moe(probe, false);
    expect(base_need > 0 && base_need < 8, "synthetic model needs some but not all offload");
    // Slack at the baseline need: shrink every budget until the need rises.
    size_t slack = 0;
    for (size_t cut = 64 * MiB; cut < 20 * GiB; cut += 64 * MiB)
        if (min_cpu_moe(synthetic(8, 2, tight, 2 * GiB + cut, 0), false) > base_need) { slack = cut; break; }
    expect(slack > 0, "calibration found the next offload step");
    // (2) 1024 needs one more host layer, 512 does not -> 512.
    {
        std::vector<split_costs> costs = {
            synthetic(8, 2, tight, 2 * GiB + slack, 0), synthetic(8, 2, tight, 2 * GiB + slack / 4, 0),
            synthetic(8, 2, tight, 2 * GiB, 0)};
        auto c = candidates_for(costs, -1);
        auto choice = choose_ubatch(c, 8, -1, false);
        expect(c[0].offload.need_cpu_moe == base_need + 1 && c[1].offload.need_cpu_moe == base_need,
            "(2) fixture: 1024 needs one more");
        expect(choice.index == 1, "(2) one more host layer at 1024 picks 512: " + choice.reason);
        expect(choice.reason.find("1024 would need") != std::string::npos, "(2) the log says why 1024 was not taken");
        std::printf("ONE_MORE_AT_1024 base=%d 1024=%d -> %d (%s)\n", base_need, c[0].offload.need_cpu_moe,
            c[(size_t) choice.index].ubatch, choice.reason.c_str());
    }
    // (3) Both wider widths need more -> 256.
    {
        std::vector<split_costs> costs = {
            synthetic(8, 2, tight, 2 * GiB + slack, 0), synthetic(8, 2, tight, 2 * GiB + slack, 0),
            synthetic(8, 2, tight, 2 * GiB, 0)};
        auto c = candidates_for(costs, -1);
        auto choice = choose_ubatch(c, 8, -1, false);
        expect(choice.index == 2 && !choice.refused, "(3) both wider need more picks 256: " + choice.reason);
        expect(choice.reason.find("1024 would need") != std::string::npos &&
               choice.reason.find("512 would need") != std::string::npos, "(3) the log names both declined widths");
        std::printf("BOTH_WIDER_NEED_MORE -> %d (%s)\n", c[(size_t) choice.index].ubatch, choice.reason.c_str());
    }
    // (4) An explicit --n-cpu-moe the run pays anyway admits the wider width.
    {
        std::vector<split_costs> costs = {
            synthetic(8, 2, tight, 2 * GiB + slack, 0), synthetic(8, 2, tight, 2 * GiB + slack / 4, 0),
            synthetic(8, 2, tight, 2 * GiB, 0)};
        auto c = candidates_for(costs, base_need + 1);
        auto choice = choose_ubatch(c, 8, base_need + 1, false);
        expect(choice.index == 0, "(4) --n-cpu-moe covering 1024's need picks 1024: " + choice.reason);
        auto exact = choose_ubatch(candidates_for(costs, base_need), 8, base_need, false);
        expect(exact.index == 1, "(4) --n-cpu-moe equal to 256's need still declines 1024: " + exact.reason);
    }
    // (5) An explicit --n-cpu-moe below the 256 need is refused, naming that need.
    {
        std::vector<split_costs> costs = {
            synthetic(8, 2, tight, 2 * GiB + slack, 0), synthetic(8, 2, tight, 2 * GiB + slack / 4, 0),
            synthetic(8, 2, tight, 2 * GiB, 0)};
        const int requested = base_need - 1;
        auto c = candidates_for(costs, requested);
        auto choice = choose_ubatch(c, 8, requested, false);
        expect(choice.refused && choice.index == 2, "(5) a request below the 256 need refuses at 256");
        const std::string message = not_enough_vram_message(80.0, 70.0, 2, requested > 0,
            c[(size_t) choice.index].offload.need_cpu_moe, 8.0 * base_need);
        const std::string phrase = "Re-run with --n-cpu-moe " + std::to_string(base_need) + " ";
        expect(message.find(phrase) != std::string::npos, "(5) the refusal names the 256 need: " + message);
        std::printf("REFUSAL requested=%d %s", requested, message.c_str());
    }
    // (6) Does not fit at all: the baseline, refused.
    {
        std::vector<split_costs> costs = {
            synthetic(8, 2, 4 * GiB, 3 * GiB, 0), synthetic(8, 2, 4 * GiB, 3 * GiB, 0), synthetic(8, 2, 4 * GiB, 3 * GiB, 0)};
        auto c = candidates_for(costs, -1);
        auto choice = choose_ubatch(c, 8, -1, false);
        expect(c[2].offload.need_cpu_moe == 9 && choice.refused && choice.index == 2, "(6) no fit refuses at 256");
    }
    // (7) An explicit width is one candidate and is never replaced, even when a
    // wider or narrower width would need less offload.
    {
        std::vector<split_costs> costs = {synthetic(8, 2, tight, 2 * GiB + slack, 0)};
        std::vector<ubatch_candidate> c(1);
        c[0].ubatch = 2048;
        c[0].offload = plan_offload(costs[0], false, -1);
        auto choice = choose_ubatch(c, 8, -1, false);
        expect(choice.index == 0 && !choice.refused, "(7) an explicit width is kept");
    }
}

static void check_engram_rules()
{
    // Two 20 GiB tables on layers 1 and 5 of the synthetic model.
    auto with_tables = [](size_t free_b, size_t held) {
        split_costs c = synthetic(8, 2, free_b, held, 0);
        c.layer_engram_bytes[1] = 20 * GiB;
        c.layer_engram_bytes[5] = 20 * GiB;
        return c;
    };
    // Roomy devices: tables fit at every width and cost no offload.
    {
        std::vector<split_costs> costs = {with_tables(90 * GiB, 3 * GiB), with_tables(90 * GiB, 2 * GiB + GiB / 2),
            with_tables(90 * GiB, 2 * GiB)};
        auto c = candidates_for(costs, -1, true);
        expect(c[2].offload.engram_device && c[2].offload.engram_refused == nullptr, "roomy devices keep the tables");
        expect(choose_ubatch(c, 8, -1, false).index == 0, "roomy devices pick 1024");
    }
    // Find a budget where the tables fit at 256 but not at a wider reserve.
    size_t step = 0;
    split_costs base = with_tables(68 * GiB, 2 * GiB);
    offload_plan base_plan = plan_offload(base, true, -1);
    if (base_plan.engram_device)
        for (size_t cut = 256 * MiB; cut < 30 * GiB; cut += 256 * MiB)
        {
            const offload_plan p = plan_offload(with_tables(68 * GiB, 2 * GiB + cut), true, -1);
            if (!p.engram_device && p.need_cpu_moe == base_plan.need_cpu_moe) { step = cut; break; }
        }
    expect(step > 0, "calibration found a width where the tables move to the host without extra offload");
    if (step > 0)
    {
        std::vector<split_costs> costs = {with_tables(68 * GiB, 2 * GiB + step), with_tables(68 * GiB, 2 * GiB),
            with_tables(68 * GiB, 2 * GiB)};
        auto c = candidates_for(costs, -1, true);
        auto choice = choose_ubatch(c, 8, -1, false);
        expect(choice.index == 1, "a width that moves the Engram tables to the host is declined: " + choice.reason);
        expect(choice.reason.find("Engram") != std::string::npos, "the log says the tables were the reason");
        std::printf("ENGRAM_DEVICE kept at 256 -> %d (%s)\n", c[(size_t) choice.index].ubatch, choice.reason.c_str());
        auto forced = choose_ubatch(c, 8, -1, true);
        expect(forced.index == 1, "TS_DSV41_ENGRAM_DEVICE=1 also declines the width that cannot hold the tables");
    }
    // The loader's own refusal reasons are unchanged.
    offload_plan none = plan_offload(with_tables(12 * GiB, 2 * GiB), true, -1);
    expect(none.engram_refused && std::string(none.engram_refused).find("do not fit these devices") != std::string::npos,
        "tables that never fit say so");
}

// ---------------------------------------------------------------------------
// DeepSeek V4.1 Flash Q4_K_M (11 shards), sizes read from the GGUF headers:
// weights except Engram tables, routed experts, Engram tables, per layer.
// ---------------------------------------------------------------------------
static const size_t Q4KM_LAYER_BYTES[] = {
    8910676440ull, 8999173080ull, 8916706264ull, 8910676440ull, 8910676440ull,
    7739783640ull, 7739783640ull, 8910676440ull, 7745813464ull, 7739783640ull,
    8910676440ull, 7739783640ull, 7739783640ull, 8910676440ull, 7834310104ull,
    7739783640ull, 8910676440ull, 7739783640ull, 7739783640ull, 8910676440ull,
    7744338904ull, 7739783640ull, 8910676440ull, 7739783640ull, 7742824920ull,
    8910676440ull, 7739783640ull, 7739783640ull, 8913717720ull, 7739783640ull,
    7739783640ull, 8910676440ull, 7742824920ull, 7739783640ull, 8910676440ull,
    8910676440ull, 8913717720ull, 8910676440ull, 8910676440ull, 8910676440ull,
};
static const size_t Q4KM_LAYER_EXPS_BYTES[] = {
    8811970560ull, 8811970560ull, 8811970560ull, 8811970560ull, 8811970560ull,
    7644119040ull, 7644119040ull, 8811970560ull, 7644119040ull, 7644119040ull,
    8811970560ull, 7644119040ull, 7644119040ull, 8811970560ull, 7644119040ull,
    7644119040ull, 8811970560ull, 7644119040ull, 7644119040ull, 8811970560ull,
    7644119040ull, 7644119040ull, 8811970560ull, 7644119040ull, 7644119040ull,
    8811970560ull, 7644119040ull, 7644119040ull, 8811970560ull, 7644119040ull,
    7644119040ull, 8811970560ull, 7644119040ull, 7644119040ull, 8811970560ull,
    8811970560ull, 8811970560ull, 8811970560ull, 8811970560ull, 8811970560ull,
};
static constexpr size_t Q4KM_ROOT_BYTES = 915322880ull, Q4KM_EMBD_BYTES = 372326400ull;
static constexpr size_t Q4KM_ENGRAM_L1 = 55296888192ull, Q4KM_ENGRAM_L14 = 55298402208ull;

static cache_geometry q4km_geometry(int64_t n_ctx)
{
    cache_geometry g;
    g.v41 = true;
    g.n_embd_head = 512;
    g.indexer_head_size = 128;
    g.n_csa_rows = pad(n_ctx / 2 + 1, 256);
    g.n_hca_rows = pad(n_ctx / 1 + 1, 256);
    g.rewind_cp = true;
    g.compress_ratios.assign(40, 0);
    for (int il = 2; il < 20; ++il) g.compress_ratios[(size_t) il] = 2;
    for (int il = 20; il < 40; ++il) g.compress_ratios[(size_t) il] = 1;
    g.kv_sources = {2, 8, 14, 20};
    return g;
}

// The seven-A40 lane (gpu7-ncm6-plain, context 65,536, ubatch 512, --n-cpu-moe 6)
// logged each device's layer range and free memory after the weight upload,
// before any slot or graph allocation:
//   device 0: layers 0..10, 4.8 GiB   device 4: layers 26..30, 6.9 GiB
//   device 1: layers 11..15, 6.8 GiB  device 5: layers 31..35, 4.7 GiB
//   device 2: layers 16..20, 5.8 GiB  device 6: layers 36..39, 10.3 GiB
//   device 3: layers 21..25, 5.8 GiB
// Free memory before the load is that plus what the upload placed there (the
// device's weights, its fixed residents and the rope tables). The values come
// out at 45,091-45,123 MiB, i.e. an A40's 45,498 MiB less its CUDA context;
// the 0.1 GiB rounding of the log leaves about +/-51 MiB of uncertainty.
static std::vector<size_t> lane_free_before_load()
{
    const int first[] = {0, 11, 16, 21, 26, 31, 36}, last[] = {10, 15, 20, 25, 30, 35, 39};
    const double free_after_gib[] = {4.8, 6.8, 5.8, 5.8, 6.9, 4.7, 10.3};
    const size_t rope = 2 * 64 * 65536 * 4;
    std::vector<size_t> out(7);
    for (int d = 0; d < 7; ++d)
    {
        size_t placed = rope;
        for (int il = first[d]; il <= last[d]; ++il)
            placed += Q4KM_LAYER_BYTES[il] - (il < 6 ? Q4KM_LAYER_EXPS_BYTES[il] : 0);
        if (d == 0) placed += Q4KM_EMBD_BYTES;
        if (d == 6) placed += Q4KM_ROOT_BYTES - Q4KM_EMBD_BYTES;
        out[(size_t) d] = (size_t) std::llround(free_after_gib[d] * double(GiB)) + placed;
    }
    return out;
}

static split_costs q4km_costs(int ubatch, int64_t n_ctx, const std::vector<size_t> & free_before, reserve_estimate & r)
{
    const cache_geometry g = q4km_geometry(n_ctx);
    const int64_t ring = ring_rows(128, ubatch);
    split_costs c;
    c.layer_bytes.assign(std::begin(Q4KM_LAYER_BYTES), std::end(Q4KM_LAYER_BYTES));
    c.layer_exps_bytes.assign(std::begin(Q4KM_LAYER_EXPS_BYTES), std::end(Q4KM_LAYER_EXPS_BYTES));
    c.layer_engram_bytes.assign(40, 0);
    c.layer_engram_bytes[1] = Q4KM_ENGRAM_L1;
    c.layer_engram_bytes[14] = Q4KM_ENGRAM_L14;
    for (int il = 0; il < 40; ++il) c.layer_cache_bytes.push_back(layer_cache_bytes(g, il, ring));
    c.fixed_bytes.assign(7, 0);
    c.fixed_bytes[0] = Q4KM_EMBD_BYTES;
    c.fixed_bytes[6] = Q4KM_ROOT_BYTES - Q4KM_EMBD_BYTES;
    r = estimate_reserve(ubatch, n_ctx, true, 5120, 4, 4, -1);
    const size_t held = r.reserve_mb * MiB + (size_t) (2 * 64 * n_ctx * 4);
    for (size_t free_b : free_before) c.dev_budget.push_back(free_b > held ? free_b - held : 0);
    return c;
}

static void check_q4km_seven_a40()
{
    const std::vector<size_t> free_before = lane_free_before_load();
    for (size_t free_b : free_before)
        expect(free_b > 44900 * MiB && free_b < 45300 * MiB, "back-derived free memory is an A40 less its context");

    // The lane's own numbers: ubatch 512 logged ring=768 and a 2,588 MiB
    // reserve (indexer 384 x1.25 + activations 60 + 2048), and it needed 6.
    expect(ring_rows(128, 512) == 768, "ring at 512 matches the lane log");
    reserve_estimate r512;
    const split_costs lane = q4km_costs(512, 65536, free_before, r512);
    expect(r512.reserve_mb == 2588 && std::lround(r512.idx_mb) == 384 && std::lround(r512.act_mb) == 60,
        "reserve at 512 matches the lane log");
    size_t gpu_weights = Q4KM_ROOT_BYTES;
    for (int il = 0; il < 40; ++il) gpu_weights += Q4KM_LAYER_BYTES[il] - (il < 6 ? Q4KM_LAYER_EXPS_BYTES[il] : 0);
    expect(std::lround(double(gpu_weights) / double(GiB) * 10) == 2630, "6 host layers leave the logged 263.0 GiB");
    const offload_plan lane_plan = plan_offload(lane, true, 6);
    expect(lane_plan.need_cpu_moe == 6, "the lane's width needs the 6 host layers it logged");
    expect(!lane_plan.engram_device && lane_plan.engram_refused &&
        std::string(lane_plan.engram_refused).find("even with every routed expert on the host") != std::string::npos,
        "the Engram tables stay host mappings for the logged reason");

    for (int64_t n_ctx : {int64_t(65536), int64_t(131072)})
    {
        std::vector<split_costs> costs;
        std::vector<reserve_estimate> reserves(3);
        const std::vector<int> widths = ubatch_candidates(UBATCH_AUTO, true, 0);
        for (size_t i = 0; i < widths.size(); ++i) costs.push_back(q4km_costs(widths[i], n_ctx, free_before, reserves[i]));
        for (int requested : {6, -1})
        {
            auto c = candidates_for(costs, requested, true);
            auto choice = choose_ubatch(c, 40, requested, false);
            std::printf("Q4KM_7xA40 n_ctx=%lld --n-cpu-moe %d: need 1024=%d 512=%d 256=%d reserve 1024=%zu 512=%zu 256=%zu MiB "
                "-> prefill ubatch: %d (auto; %s)%s\n",
                (long long) n_ctx, requested, c[0].offload.need_cpu_moe, c[1].offload.need_cpu_moe,
                c[2].offload.need_cpu_moe, reserves[0].reserve_mb, reserves[1].reserve_mb, reserves[2].reserve_mb,
                c[(size_t) std::max(0, choice.index)].ubatch, choice.reason.c_str(), choice.refused ? " REFUSED" : "");
            expect(choice.refused || c[(size_t) choice.index].offload.need_cpu_moe <= std::max(requested, c[2].offload.need_cpu_moe),
                "the chosen width never costs an extra host layer");
            if (n_ctx == 65536)
            {
                expect(!choice.refused && c[(size_t) choice.index].ubatch == 1024 &&
                       c[(size_t) choice.index].offload.need_cpu_moe == 6,
                    "the 7-A40 lane at 65,536 context runs 1024 with 6 CPU-MoE layers");
            }
        }
        if (n_ctx == 65536)
        {
            auto c = candidates_for(costs, 5, true);
            auto choice = choose_ubatch(c, 40, 5, false);
            const std::string message = not_enough_vram_message(263.0, 315.0, 7, true,
                c[(size_t) choice.index].offload.need_cpu_moe, 48.2);
            expect(choice.refused && choice.index == 2 && message.find("Re-run with --n-cpu-moe 6 ") != std::string::npos,
                "--n-cpu-moe 5 on the lane is refused naming 6");
        }
    }
}

int main()
{
    check_candidate_lists();
    check_synthetic_choices();
    check_engram_rules();
    check_q4km_seven_a40();
    if (failures)
    {
        std::fprintf(stderr, "%d ubatch plan check(s) failed\n", failures);
        return 1;
    }
    std::printf("DSV4_UBATCH_PLAN passed\n");
    return 0;
}
