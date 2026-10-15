// Three real quantized type groups share one physical ceiling. Included in the
// CUDA lifecycle fixture; this is operator/ownership evidence, not model quality.
#ifdef TSG_GGML_USE_CUDA
void glm_group_quota_checks() {
    constexpr int count = 64, top = 8, maximum_rows = 128, decode_rows = 3;
    tsg::host_moe_expert_cache_release();
    std::array<Weights, 3> weights;
    weights[0].types = {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0};
    weights[1].types = {GGML_TYPE_IQ1_M, GGML_TYPE_IQ1_M, GGML_TYPE_IQ4_NL};
    weights[2].types = {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ4_NL};
    std::vector<tsg::HostMoeGroupMinimum> minima;
    std::array<std::map<int, std::unique_ptr<FullResidentGraph>>, 3> oracles;
    std::uint64_t planning_token = 0;
    require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(64LL << 20,
        maximum_rows, 3, decode_rows, &planning_token) == 1, "Cannot begin size-only group plan");
    struct PlanningCleanup {
        std::uint64_t& token;
        ~PlanningCleanup() { if (token) { std::int64_t released = 0; TSGgml_EndHostMoeExpertCacheScope(token, &released); } }
    } planning_cleanup{planning_token};
    for (int group = 0; group < 3; ++group) {
        auto& w = weights[group]; w.fill(.7f + group * .4f, count, top);
        w.segment.layer = group; w.segment.pooled_model_experts = count;
        std::size_t minimum = 0;
        require(tsg::host_moe_expert_cache_minimum_prefill_plan(w.segment, decode_rows,
            maximum_rows, g_backend, minimum), "Cannot measure exact group minimum");
        minima.push_back({tsg::host_moe_expert_group_key(w.segment), minimum});
        for (int rows : {1, 20, 32, 64, 128})
            oracles[group].emplace(rows, std::make_unique<FullResidentGraph>(w, rows));
    }
    std::int64_t planning_released = 0;
    require(TSGgml_EndHostMoeExpertCacheScope(planning_token, &planning_released) == 1
        && planning_released == 0, "Metadata-only group planning retained cache payload");
    planning_token = 0;
    tsg::HostMoeGroupQuotaPlan plan(minima);
    const auto budget = plan.minimum() + 17; // Exercise deterministic byte remainder.
    const auto largest = (std::max)({minima[0].bytes, minima[1].bytes, minima[2].bytes});
    require(budget < largest * 3, "Fixture does not distinguish equal and measured-group quotas");
    SharedLedger ledger; ledger.capacity = static_cast<std::int64_t>(budget);
    ledger.host_capacity = 65536; ledger.multiple_cache_entries = true;
    require(ledger.attach(true), "Cannot attach actual three-group ownership ledger");
    std::uint64_t token = 0;
    const auto path = fixture_file_path("glm-measured-groups-");
    struct Cleanup {
        std::string path; std::uint64_t& token;
        ~Cleanup() {
            std::int64_t released = 0;
            if (token) TSGgml_EndHostMoeExpertCacheScope(token, &released);
            tsg::host_moe_expert_cache_release();
            tsg::host_moe_expert_cache_test_file_slice_limit(32u << 20);
            std::remove(path.c_str());
        }
    } cleanup{path, token};
    std::int64_t offsets[3][3]{};
    {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        require(bool(file), "Cannot create three-group source");
        for (int group = 0; group < 3; ++group) for (int part = 0; part < 3; ++part) {
            file.write("group-offset", 12); offsets[group][part] = static_cast<std::int64_t>(file.tellp());
            const auto& bytes = weights[group].quantized[part];
            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }
        require(bool(file), "Cannot write three-group source");
    }
    for (int group = 0; group < 3; ++group) for (int part = 0; part < 3; ++part) {
        auto& bytes = weights[group].quantized[part];
        require(TSGgml_RegisterHostFileSource(bytes.data(), bytes.size(), path.c_str(), offsets[group][part]) == 1,
            "Cannot register three-group immutable extent");
        std::fill(bytes.begin(), bytes.end(), 0); // GPU oracle owns original bytes.
    }
    tsg::host_moe_expert_cache_test_file_slice_limit(65536);
    require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(budget,
        maximum_rows, 3, decode_rows, &token) == 1, "Cannot begin three-group scope");
    std::int64_t released = 0;
    require(TSGgml_UpdateHostMoeExpertCacheScope(token, plan.minimum() - 1, &released) == 1
        && released == 0, "Cannot lower an empty scope below its complete group floor");
    require(!tsg::host_moe_expert_cache_install_group_plan(minima, maximum_rows, decode_rows, g_backend)
        && ledger.allocations.empty(), "Insufficient total floor installed a partial plan");
    require(TSGgml_UpdateHostMoeExpertCacheScope(token, budget, &released) == 1,
        "Cannot recover complete group floor");
    require(!tsg::host_moe_expert_cache_install_group_plan(minima, maximum_rows - 1, decode_rows, g_backend),
        "Mismatching prefill shape installed a plan");
    auto duplicate = minima; duplicate[1] = duplicate[0];
    bool rejected_duplicate = false;
    try { rejected_duplicate = !tsg::host_moe_expert_cache_install_group_plan(duplicate, maximum_rows, decode_rows, g_backend); }
    catch (const std::invalid_argument&) { rejected_duplicate = true; }
    require(rejected_duplicate && tsg::host_moe_expert_cache_install_group_plan(minima, maximum_rows, decode_rows, g_backend)
        && !tsg::host_moe_expert_cache_install_group_plan(minima, maximum_rows, decode_rows, g_backend),
        "Plan publication is not atomic and exclusive");
    std::size_t compared = 0;
    auto run = [&](int group, int rows, bool expected_success) {
        std::vector<float> input(rows * hidden), routes(rows * top), actual(rows * hidden, -999.0f), expected;
        std::vector<std::int32_t> ids(rows * top);
        for (int row = 0; row < rows; ++row) {
            for (int h = 0; h < hidden; ++h) input[row * hidden + h] = std::cos(h * .041f + row * .31f);
            for (int k = 0; k < top; ++k) {
                ids[row * top + k] = (row * top + k) % count;
                routes[row * top + k] = (k + 1.0f) / 36.0f;
            }
        }
        if (rows <= 8) {
            expected.resize(actual.size());
            for (int row = 0; row < rows; ++row) {
                // Use the oracle's runtime-checked vector interface. A local
                // constexpr captured by this lambda is not a portable MSVC
                // std::array extent; the graph still executes exactly one row.
                std::vector<float> x(hidden), probability(top);
                std::vector<std::int32_t> selected(top);
                std::copy_n(input.begin() + row * hidden, hidden, x.begin());
                std::copy_n(routes.begin() + row * top, top, probability.begin());
                std::copy_n(ids.begin() + row * top, top, selected.begin());
                const auto answer = oracles[group].at(1)->compute_batch(x, selected, probability);
                std::copy(answer.begin(), answer.end(), expected.begin() + row * hidden);
            }
        } else expected = oracles[group].at(rows)->compute_batch(input, ids, routes);
        auto hm = weights[group].segment; hm.seq_len = rows; hm.independent_decode_rows = rows == 3;
        const auto ok = tsg::host_moe_cached_experts(hm, input.data(), ids.data(), routes.data(), actual.data(),
            rows > 8 ? "glm5-next bulk expert prefill" : "glm5-next selected expert seam", g_backend);
        require((ok == 1) == expected_success, "Three-group execution/refusal mismatch");
        if (expected_success) {
            require(std::all_of(actual.begin(), actual.end(), [](float x) { return std::isfinite(x); })
                && std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
                "Group quota or changing wide scratch changed complete CUDA output");
            compared += actual.size();
        } else require(std::all_of(actual.begin(), actual.end(), [](float x) { return x == -999.0f; }),
            "Refused group partially published output");
        require(ledger.valid && ledger.pending == 0 && ledger.committed <= static_cast<std::int64_t>(budget),
            "Three-group transaction exceeded shared credit or lost physical ownership");
    };
    run(0, 128, true);
    const auto first_owned = ledger.committed;
    ledger.fail_commit = true; run(1, 128, false); ledger.fail_commit = false;
    require(ledger.committed == first_owned, "Failed second group released its neighbour's credit");
    run(0, 128, true); run(1, 128, true); run(2, 128, true);
    for (int rows : {64, 20, 32, 3, 1, 128}) for (int group = 0; group < 3; ++group) run(group, rows, true);
    std::array<std::int64_t, 24> entries{}; int entry_count = 0;
    require(TSGgml_HostMoeExpertCacheEntryStats(entries.data(), 3, &entry_count) == 1 && entry_count == 3,
        "Three simultaneously live type groups were not retained");
    require(TSGgml_UpdateHostMoeExpertCacheScope(token, plan.minimum() - 1, &released) == 1
        && ledger.committed == 0 && ledger.pending == 0, "Sub-floor resize retained unusable group payloads");
    const auto reservations = ledger.reserve_calls; run(0, 1, false);
    require(ledger.reserve_calls == reservations, "Missing full floor attempted a partial allocation");
    require(TSGgml_UpdateHostMoeExpertCacheScope(token, budget, &released) == 1, "Cannot restore measured floors");
    for (int group = 0; group < 3; ++group) run(group, 3, true);
    require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1, "Three-group scope failed cleanup");
    token = 0; ledger.check_empty(); require(ledger.detach(), "Three-group callbacks retained an owner");
    for (auto& w : weights) for (auto& bytes : w.quantized) tsg::host_moe_expert_cache_on_drop(bytes.data());
    require(std::remove(path.c_str()) == 0, "Three-group file handles survived unregister");
    require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(budget, maximum_rows, 3, decode_rows, &token) == 1
        && tsg::host_moe_expert_cache_install_group_plan(minima, maximum_rows, decode_rows, g_backend),
        "Ended scope leaked its group metadata into a fresh session");
    require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1, "Fresh empty group scope did not close");
    token = 0;
    std::printf("PASS: actual 3-group measured minima=%zu budget=%zu below equal=%zu; %zu finite CUDA floats byte-exact, widths128/64/20/32/3/1, poisoned file identities, failed second-group commit isolation, shrink/regrow, physical-before-credit cleanup\n",
        plan.minimum(), budget, largest * 3, compared);
}
#endif
