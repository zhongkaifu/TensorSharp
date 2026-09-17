// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

// DeepSeek V4 / V4.1 load planning that depends only on byte counts: what a
// layer costs its device, how layers pack into the per-device budgets, how
// many leading layers must move their routed experts to the host, and -- for
// an automatic prefill width -- which ubatch to run.
//
// Kept free of ggml and of the model struct so tests/dsv4_ubatch_plan_test.cpp
// can drive it with synthetic budgets and with a real checkpoint's sizes.
// ggml_ops_deepseek4.cpp's dsv4_load is the only production caller.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace tsg_dsv4_plan {

// n_ubatch the managed side passes when it wants the loader to choose. 0
// already means "512" (and the parity fixtures pass explicit widths), so the
// sentinel is distinct from both.
constexpr int UBATCH_AUTO = -1;

// Widths the automatic choice evaluates, widest first; the last one is the
// baseline whose routed-expert offload bounds the others. 2048 is not a
// candidate: the graph reserve below was validated only up to 1024 (a
// 57,424-token prefill at 1024 peaked with 1,522 MiB free against a 3,072 MiB
// reserve on eight A40s), and a ggml device OOM is a segfault, not an error.
constexpr int AUTO_UBATCH_CANDIDATES[] = {1024, 512, 256};

// Candidates for a model whose DSpark drafter (block size `dspark_block`, 0
// without one) needs ubatch >= block + 1 so a verify fits one micro-batch.
inline std::vector<int> auto_ubatch_candidates(int dspark_block)
{
    std::vector<int> out;
    for (int ubatch : AUTO_UBATCH_CANDIDATES)
        if (dspark_block <= 0 || ubatch > dspark_block) out.push_back(ubatch);
    if (out.empty()) out.push_back(dspark_block + 1);
    return out;
}

// The widths a load evaluates. An explicit request (anything but UBATCH_AUTO)
// is the only candidate, so it is never changed: 0 or less keeps meaning 512.
// UBATCH_AUTO evaluates the automatic candidates on accelerators and keeps the
// executor's 256 on the CPU device.
inline std::vector<int> ubatch_candidates(int requested, bool accelerators, int dspark_block)
{
    if (requested != UBATCH_AUTO) return { requested > 0 ? requested : 512 };
    if (!accelerators) return { 256 };
    return auto_ubatch_candidates(dspark_block);
}

inline int64_t pad(int64_t v, int64_t p) { return (v + p - 1) / p * p; }

// Raw sliding-window ring rows for one width: the window plus one ubatch.
inline int64_t ring_rows(int64_t n_swa, int64_t n_ubatch) { return pad(n_swa + n_ubatch, 256); }

// ---------------------------------------------------------------------------
// Per-layer resident caches
// ---------------------------------------------------------------------------
struct cache_geometry
{
    bool v41 = false;
    int64_t n_embd_head = 0;         // attention.key_length
    int64_t indexer_head_size = 0;
    int64_t n_csa_rows = 0, n_hca_rows = 0;
    int64_t state_extra = 0;         // DSpark block size when a drafter is loaded
    bool rewind_cp = false;          // V4.1 rewind-checkpoint shadows
    std::vector<int32_t> compress_ratios;
    std::vector<int32_t> kv_sources; // V4.1 layers that own a shared compressed cache
    int64_t csa_ratio = 4, hca_ratio = 128;   // plain V4
};

// KV caches and compressor state rings a layer allocates on its own device
// right after its weights. The split has to price them: a device packed to the
// last byte of weights fails at slot allocation instead of at load, which reads
// as a runtime crash rather than a capacity problem.
inline size_t layer_cache_bytes(const cache_geometry & g, int il, int64_t ring_raw)
{
    const int64_t head = g.n_embd_head;
    const int64_t idx = g.indexer_head_size;
    size_t b = (size_t) (head * ring_raw * 2);                          // raw_k F16
    const int ratio = g.compress_ratios[(size_t) il];
    if (g.v41)
    {
        const bool owns = std::find(g.kv_sources.begin(), g.kv_sources.end(), il) != g.kv_sources.end();
        if (owns)
        {
            const int64_t rows = ratio == 2 ? g.n_csa_rows : g.n_hca_rows;
            b += (head + idx) * rows * 2;
            if (ratio > 1) b += 2 * head * (ratio + g.state_extra + 1) * 4;
        }
        if (g.rewind_cp)
        {
            // Rewind-checkpoint shadows: the raw ring, and the compressor
            // state ring where the layer owns one.
            b += (size_t) (head * ring_raw * 2);
            if (ratio > 1 && owns) b += (size_t) (2 * head * (ratio + g.state_extra + 1) * 4);
        }
    }
    else if (ratio == g.csa_ratio)
    {
        const int64_t st = 2 * g.csa_ratio + g.state_extra + 1;
        b += (size_t) (head * g.n_csa_rows * 2);                        // csa_k F16
        b += (size_t) (idx * g.n_csa_rows * 2);                         // lid_k F16
        b += (size_t) (2 * (2 * head) * st * 4);                        // comp_state kv+score F32
        b += (size_t) (2 * (2 * idx) * st * 4);                         // lid_state  kv+score F32
    }
    else if (ratio == g.hca_ratio)
    {
        const int64_t st = g.hca_ratio + g.state_extra + 1;
        b += (size_t) (head * g.n_hca_rows * 2);                        // hca_k F16
        b += (size_t) (2 * head * st * 4);                              // comp_state kv+score F32
    }
    return b + 13 * 256;   // ggml buffer alignment padding per tensor
}

// ---------------------------------------------------------------------------
// Device memory held back for the graph
// ---------------------------------------------------------------------------
struct reserve_estimate
{
    size_t reserve_mb = 0;
    double idx_mb = 0, act_mb = 0;
    bool overridden = false;
};

// The scheduler's compute buffers are sized from the largest ubatch graph and
// cannot be known before the model exists, so the split holds back a reserve.
// A flat 2 GiB was too small for this architecture once the weights nearly fill
// the cards: the lightning indexer's top-k runs an argsort over every visible
// compressed row, and CUB takes its workspace from the CUDA VMM pool at RUN
// time, not from any layer's budget. A 1024-token ubatch over a 64k context is
// ~768 MiB for that one transient alone, and a DeepSeek V4.1 Q4_K_M prefill of a
// 28k-token prompt aborted in argsort_f32_i32_cuda_cub with the flat reserve.
//
// So price the transients that scale: the indexer's scores and its sort
// workspace, and the hidden activations of one ubatch across the streams.
// TS_DSV4_VRAM_RESERVE_MB (`override_mb`, negative when unset) replaces it,
// including downward, and upward for a rig that wants more margin.
//
// The factor on the indexer term used to be 4, because the reserve also had to
// cover a graph cache capped by ENTRY COUNT: twelve prefill graphs of a few
// hundred MiB each is more than any headroom. dsv4_trim_graph_cache now bounds
// that cache by bytes, so the reserve covers one graph's compute buffers plus
// the run-time transients that never pass through a graph buffer (ggml-cuda
// takes CUB's segmented argsort workspace straight from the VMM pool).
//
// Measured on the eight-A40 VM, Q4_K_M, ubatch 1024, 64k context: the largest
// graph's compute buffers were 1,336 MiB on a device (1.7x the indexer term),
// and a 57,424-token prefill peaked with 1,522 MiB still free on the tightest
// device against a 3,072 MiB reserve. 1.25 lands just above that; holding back
// more costs routed-expert offload -- 5,240 MiB forced three CPU-MoE layers
// where 3,174 MiB needs one, worth 350 -> 480 prefill tok/s.
inline reserve_estimate estimate_reserve(int64_t n_ubatch, int64_t n_ctx, bool v41, int64_t n_embd, int64_t hc_mult,
                                         int64_t csa_ratio, long override_mb)
{
    reserve_estimate r;
    const int64_t comp_rows = n_ctx / (v41 ? 1 : csa_ratio) + 1;
    r.idx_mb = (double) n_ubatch * comp_rows * 4.0 * 3.0 / (1024.0 * 1024.0);
    r.act_mb = (double) n_ubatch * n_embd * 4.0 * (hc_mult + 2) / (1024.0 * 1024.0);
    r.reserve_mb = (size_t) std::max(2048.0, 1.25 * r.idx_mb + r.act_mb + 2048.0);
    if (override_mb >= 0)
    {
        r.reserve_mb = (size_t) override_mb;
        r.overridden = true;
    }
    return r;
}

// ---------------------------------------------------------------------------
// Layer -> device split and routed-expert offload
// ---------------------------------------------------------------------------
struct split_costs
{
    std::vector<size_t> layer_bytes;          // per layer: weights except the Engram table
    std::vector<size_t> layer_exps_bytes;     // per layer: routed experts (what --n-cpu-moe moves)
    std::vector<size_t> layer_engram_bytes;   // per layer: Engram table, priced only when device-resident
    std::vector<size_t> layer_cache_bytes;    // per layer: resident caches for this width
    std::vector<std::vector<size_t>> tp_bytes;// [layer][rank] routed-MoE strips when tp_ranks > 0
    int tp_ranks = 0;
    std::vector<size_t> fixed_bytes;          // per device: embedding, output head, drafter and its rings
    std::vector<size_t> dev_budget;           // per device: free memory minus this width's reserve

    int n_layer() const { return (int) layer_bytes.size(); }
    int n_dev() const { return (int) dev_budget.size(); }
};

inline size_t layer_cost(const split_costs & c, int il, int n_cpu, bool engram_device)
{
    size_t w = c.layer_bytes[(size_t) il];
    if (il < n_cpu || c.tp_ranks) w -= c.layer_exps_bytes[(size_t) il];
    if (engram_device) w += c.layer_engram_bytes[(size_t) il];
    return w + c.layer_cache_bytes[(size_t) il];
}

// Layers stay in pipeline order, so every device takes one contiguous run:
// fill each device up to `frac` of its budget, and report whether all of them
// fit. `out` (when given) receives each layer's device.
inline bool pack(const split_costs & c, bool engram_device, double frac, int n_cpu, std::vector<int> * out)
{
    const int n_gpu = c.n_dev();
    if (n_gpu <= 0) return false;
    auto fixed = c.fixed_bytes;
    if (c.tp_ranks)
    {
        for (int il = n_cpu; il < c.n_layer(); ++il)
            for (int d = 0; d < c.tp_ranks; ++d) fixed[(size_t) d] += c.tp_bytes[(size_t) il][(size_t) d];
        for (int d = 0; d < n_gpu; ++d)
            if (fixed[(size_t) d] > (size_t) (c.dev_budget[(size_t) d] * frac)) return false;
    }
    int dev = 0;
    size_t used = fixed[0];
    for (int il = 0; il < c.n_layer(); il++)
    {
        const size_t cost = layer_cost(c, il, n_cpu, engram_device);
        while (used + cost > (size_t) (c.dev_budget[(size_t) dev] * frac))
        {
            if (dev + 1 >= n_gpu) return false;
            used = fixed[(size_t) ++dev];
        }
        used += cost;
        if (out) (*out)[(size_t) il] = dev;
    }
    return true;
}

// Fewest leading layers that must give up their routed experts; n_layer + 1
// when even every expert on the host does not fit.
inline int min_cpu_moe(const split_costs & c, bool engram_device)
{
    int n = 0;
    while (n <= c.n_layer() && !pack(c, engram_device, 1.0, n, nullptr)) n++;
    return n;
}

struct offload_plan
{
    int need_cpu_moe = 0;       // n_layer + 1: does not fit
    int without_tables = 0;     // the same need with host-mapped Engram tables
    bool engram_device = false; // the tables stay on their layers' GPUs
    const char * engram_refused = nullptr;   // why a wanted device placement was declined
};

// The routed-expert offload one width needs, and whether the V4.1 Engram tables
// can stay GPU-resident at it (`want_engram_device`: V4.1 on accelerators with
// a get_rows kernel, and not TS_DSV41_ENGRAM_DEVICE=0).
//
// The tables are kept on GPUs only when two things hold: they cost no
// routed-expert offload beyond what this run was going to pay anyway (paying
// for device tables with host expert matmuls on every token is a bad trade --
// though an operator who already asked for --n-cpu-moe should not be refused a
// placement that fits inside it), and they still leave a little of every
// device's budget unspent (the packer prices ONE sequence slot's caches, and
// 60 GiB of tables would otherwise consume exactly the headroom the next
// concurrent sequence needs).
inline offload_plan plan_offload(const split_costs & c, bool want_engram_device, int n_cpu_moe_req)
{
    offload_plan p;
    const int n_layer = c.n_layer();
    p.need_cpu_moe = min_cpu_moe(c, want_engram_device);
    p.without_tables = p.need_cpu_moe;
    if (!want_engram_device) return p;

    constexpr double engram_device_margin = 0.95;
    const int with_tables = p.need_cpu_moe;
    p.without_tables = min_cpu_moe(c, false);
    const int already_paying = n_cpu_moe_req >= 0 ? std::max(n_cpu_moe_req, p.without_tables) : p.without_tables;
    if (with_tables > n_layer)
        p.engram_refused = "they do not fit these devices even with every routed expert on the host";
    else if (with_tables > already_paying)
        p.engram_refused = "they would cost routed-expert offload this run was not already paying";
    else if (!pack(c, true, engram_device_margin, std::max(with_tables, n_cpu_moe_req < 0 ? 0 : n_cpu_moe_req), nullptr))
        p.engram_refused = "they would leave no headroom for a second concurrent sequence";
    p.engram_device = p.engram_refused == nullptr;
    p.need_cpu_moe = p.engram_device ? with_tables : p.without_tables;
    return p;
}

// ---------------------------------------------------------------------------
// Automatic prefill width
// ---------------------------------------------------------------------------
struct ubatch_candidate
{
    int ubatch = 0;
    offload_plan offload;
};

struct ubatch_choice
{
    int index = 0;          // into the candidates
    bool refused = false;   // the baseline itself cannot run as requested; its need is what to report
    std::string reason;
};

// Pick the widest candidate that costs decode nothing: no more host
// routed-expert layers than the baseline (the last, narrowest candidate) --
// or than an explicit --n-cpu-moe the run pays anyway -- and no Engram table
// moved off the GPUs that the baseline keeps there. Wider prefill is cheaper
// per token (a resident routed-expert layer costs about the same per chunk at
// any width), but an extra host layer is paid on every decoded token.
//
// `n_cpu_moe_req` < 0 is automatic offload; >= 0 is the operator's count
// (0 = none). When the baseline itself needs more than an explicit count, or
// does not fit at all, the baseline is returned so the caller's refusal names
// the narrowest width's need.
inline ubatch_choice choose_ubatch(const std::vector<ubatch_candidate> & candidates, int n_layer, int n_cpu_moe_req,
                                   bool engram_device_forced)
{
    ubatch_choice choice;
    if (candidates.empty()) { choice.index = -1; choice.refused = true; choice.reason = "no candidate width"; return choice; }
    const int base_index = (int) candidates.size() - 1;
    const auto & base = candidates[(size_t) base_index];
    const std::string base_name = std::to_string(base.ubatch);
    auto layers = [](int n) { return std::to_string(n) + " routed-expert CPU layer" + (n == 1 ? "" : "s"); };

    choice.index = base_index;
    if (base.offload.need_cpu_moe > n_layer)
    {
        choice.refused = true;
        choice.reason = base_name + " does not fit even with every routed expert on the host";
        return choice;
    }
    if (n_cpu_moe_req >= 0 && n_cpu_moe_req < base.offload.need_cpu_moe)
    {
        choice.refused = true;
        choice.reason = "--n-cpu-moe " + std::to_string(n_cpu_moe_req) + " is below the " +
            layers(base.offload.need_cpu_moe) + " " + base_name + " needs";
        return choice;
    }
    if (engram_device_forced && !base.offload.engram_device)
    {
        choice.refused = true;
        choice.reason = "TS_DSV41_ENGRAM_DEVICE=1 does not fit at " + base_name;
        return choice;
    }

    const int allowed = n_cpu_moe_req >= 0 ? std::max(n_cpu_moe_req, base.offload.need_cpu_moe)
                                           : base.offload.need_cpu_moe;
    std::string declined;
    for (int i = 0; i < base_index; ++i)
    {
        const auto & c = candidates[(size_t) i];
        std::string why;
        if (c.offload.need_cpu_moe > n_layer)
            why = std::to_string(c.ubatch) + " does not fit";
        else if (c.offload.need_cpu_moe > allowed)
            why = std::to_string(c.ubatch) + " would need " + layers(c.offload.need_cpu_moe) + " against " +
                std::to_string(allowed) + (n_cpu_moe_req > base.offload.need_cpu_moe ? " requested" : " at " + base_name);
        else if (base.offload.engram_device && !c.offload.engram_device)
            why = std::to_string(c.ubatch) + " would move the Engram tables to host mappings";
        else if (engram_device_forced && !c.offload.engram_device)
            why = std::to_string(c.ubatch) + " does not fit the TS_DSV41_ENGRAM_DEVICE=1 tables";
        if (why.empty())
        {
            choice.index = i;
            break;
        }
        declined += (declined.empty() ? "" : "; ") + why;
    }
    const auto & chosen = candidates[(size_t) choice.index];
    choice.reason = std::to_string(chosen.ubatch) + " needs " + layers(chosen.offload.need_cpu_moe);
    if (choice.index != base_index)
        choice.reason += chosen.offload.need_cpu_moe == base.offload.need_cpu_moe
            ? ", the same as " + base_name : ", within " + std::to_string(allowed) + " requested";
    if (!declined.empty()) choice.reason += "; " + declined;
    return choice;
}

// The load refusal for an explicit --n-cpu-moe below what the split needs.
// Naming WHICH number would work is the whole value: the operator cannot
// derive it from the model size, because what has to fit is the weights PLUS
// this context's KV caches. Launch scripts parse "Re-run with --n-cpu-moe N",
// so that phrase is part of the contract.
inline std::string not_enough_vram_message(double weights_gib, double free_gib, int n_gpu, bool offload_requested,
                                           int need_cpu_moe, double need_host_gib)
{
    char text[640];
    std::snprintf(text, sizeof(text),
        "[dsv4] not enough VRAM: %.1f GiB of weights plus this context's KV caches against "
        "%.1f GiB free across %d device(s)%s. Re-run with --n-cpu-moe %d (moves the routed "
        "experts of the first %d layer(s), %.1f GiB, to system RAM) or --cpu-moe to offload "
        "every layer.\n",
        weights_gib, free_gib, n_gpu, offload_requested ? " at the requested offload" : "",
        need_cpu_moe, need_cpu_moe, need_host_gib);
    return text;
}

} // namespace tsg_dsv4_plan
