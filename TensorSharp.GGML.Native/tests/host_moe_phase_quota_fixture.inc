// Three real quantized type groups share one physical ceiling. Included in the
// CUDA lifecycle fixture; this is operator/ownership evidence, not model quality.
#ifdef TSG_GGML_USE_CUDA
void glm_phase_quota_checks(bool coexist = false, bool direct_only_wide = false) {
    constexpr int count = 64, top = 8, maximum_rows = 1024, decode_rows = 3;
    tsg::host_moe_expert_cache_release();
    std::array<Weights, 3> weights;
    weights[0].types = {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0};
    weights[1].types = {GGML_TYPE_IQ1_M, GGML_TYPE_IQ1_M, GGML_TYPE_IQ4_NL};
    weights[2].types = {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ4_NL};
    std::vector<tsg::HostMoePhaseGroupMinimum> minima;
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
        w.segment.layer = group; w.segment.pooled_model_experts = coexist ? count * 16 : count;
        std::size_t narrow = 0, wide = 0;
        require(tsg::host_moe_expert_cache_minimum_phase_plan(w.segment, decode_rows,
            maximum_rows, g_backend, narrow, wide), "Cannot measure exact phase group minimum");
        minima.push_back({tsg::host_moe_expert_group_key(w.segment), narrow, wide});
        for (int rows : {1, 20, 32, 64, 128, 256, 384, 512, 1024})
            oracles[group].emplace(rows, std::make_unique<FullResidentGraph>(w, rows));
    }
    std::int64_t planning_released = 0;
    require(TSGgml_EndHostMoeExpertCacheScope(planning_token, &planning_released) == 1
        && planning_released == 0, "Metadata-only group planning retained cache payload");
    planning_token = 0;
    tsg::HostMoePhaseQuotaPlan plan(minima);
    const auto all_wide = minima[0].wide_bytes + minima[1].wide_bytes + minima[2].wide_bytes;
    const auto budget = (coexist ? all_wide : plan.minimum()) + 17;
    require(coexist ? budget >= all_wide : budget < all_wide,
        "Fixture does not distinguish simultaneous-wide and pressure phases");
    SharedLedger ledger; ledger.capacity = static_cast<std::int64_t>(budget);
    ledger.host_capacity = 65536; ledger.multiple_cache_entries = true;
    require(ledger.attach(true), "Cannot attach actual three-group ownership ledger");
    std::uint64_t token = 0;
    const auto path = fixture_file_path("glm-phase-groups-");
    struct Cleanup {
        std::string path; std::uint64_t& token;
        ~Cleanup() {
            std::int64_t released = 0;
            if (token) TSGgml_EndHostMoeExpertCacheScope(token, &released);
            tsg::g_test_file_failure_countdown = 0;
            tsg::host_moe_expert_cache_test_fail_next(0);
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
    require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(budget + 1,
        maximum_rows, 3, decode_rows, &token) == 1, "Cannot begin three-group scope");
    std::int64_t released = 0;
    require(TSGgml_UpdateHostMoeExpertCacheScope(token, plan.minimum() - 1, &released) == 1,
        "Cannot lower the initial scope before phase installation");
    require(!tsg::host_moe_expert_cache_install_phase_plan(minima, maximum_rows, decode_rows, g_backend)
        && ledger.allocations.empty(), "Insufficient total floor installed a partial plan");
    require(TSGgml_UpdateHostMoeExpertCacheScope(token, budget, &released) == 1,
        "Cannot recover complete group floor");
    require(!tsg::host_moe_expert_cache_install_phase_plan(minima, maximum_rows - 1, decode_rows, g_backend),
        "Mismatching prefill shape installed a plan");
    auto duplicate = minima; duplicate[1] = duplicate[0];
    bool rejected_duplicate = false;
    try { rejected_duplicate = !tsg::host_moe_expert_cache_install_phase_plan(duplicate, maximum_rows, decode_rows, g_backend); }
    catch (const std::invalid_argument&) { rejected_duplicate = true; }
    require(rejected_duplicate && tsg::host_moe_expert_cache_install_phase_plan(minima, maximum_rows, decode_rows, g_backend)
        && !tsg::host_moe_expert_cache_install_phase_plan(minima, maximum_rows, decode_rows, g_backend),
        "Plan publication is not atomic and exclusive");
    std::size_t compared = 0;
    std::uint64_t narrow_file_reads = 0, wide_direct_reads = 0;
    DeviceInputs widest_device(maximum_rows, top);
    const auto reject_complete_shape = [&](int rows, int selected) {
        auto hm = weights[0].segment; hm.seq_len = rows; hm.n_used = selected;
        std::vector<float> input(rows * hidden), routes(rows * selected, 1.0f / selected), actual(rows * hidden, -999.0f);
        std::vector<std::int32_t> ids(rows * selected, 0);
        const auto before = Stats(); const auto reserved = ledger.reserve_calls, files_before = tsg::g_test_file_reads;
        require(tsg::host_moe_cached_experts(hm, input.data(), ids.data(), routes.data(), actual.data(),
            "glm5-next bulk expert prefill", g_backend) == 0,
            "GLM overbound row/high-topk route shape was accepted");
        const auto after = Stats();
        require(before.reserved == after.reserved && before.calls == after.calls && before.hits == after.hits
            && before.misses == after.misses && ledger.reserve_calls == reserved && tsg::g_test_file_reads == files_before
            && std::all_of(actual.begin(), actual.end(), [](float value) { return value == -999.0f; }),
            "GLM complete overbound shape read/allocated/mutated/published before refusal");
    };
    reject_complete_shape(1025, 8); reject_complete_shape(1024, 9); reject_complete_shape(129, 64);
    const auto reject_complete_device_routes = [&](bool backend_fault) {
        auto hm = weights[0].segment; hm.seq_len = maximum_rows;
        hm.moe_in = widest_device.x; hm.weights = widest_device.routes; hm.moe_out = widest_device.output;
        std::vector<float> input(maximum_rows * hidden), routes(maximum_rows * top, 1.0f / top), actual(maximum_rows * hidden, -999.0f);
        std::vector<std::int32_t> ids(maximum_rows * top, 0);
        if (!backend_fault) routes.back() = (std::numeric_limits<float>::quiet_NaN)();
        ggml_backend_tensor_set(widest_device.x, input.data(), 0, input.size() * sizeof(float));
        ggml_backend_tensor_set(widest_device.routes, routes.data(), 0, routes.size() * sizeof(float));
        ggml_backend_tensor_set(widest_device.output, actual.data(), 0, actual.size() * sizeof(float));
        const auto before = Stats(); const auto reserved = ledger.reserve_calls, files_before = tsg::g_test_file_reads;
        bool threw = false; int result = -99;
        if (backend_fault) tsg::host_moe_expert_cache_test_fail_next(5);
        try { result = tsg::host_moe_cached_experts(hm, nullptr, ids.data(), nullptr, nullptr,
            "glm5-next bulk expert prefill", g_backend); }
        catch (const std::runtime_error& error) {
            if (backend_fault && std::strcmp(error.what(), "qwen4exp expert cache: routing download failed") == 0) threw = true;
            else throw;
        }
        require(backend_fault ? threw : result == 0, "Last8192 route NaN/backend failure was accepted");
        const auto after = Stats();
        ggml_backend_tensor_get(widest_device.output, actual.data(), 0, actual.size() * sizeof(float));
        require(before.reserved == after.reserved && before.calls == after.calls && before.hits == after.hits
            && before.misses == after.misses && ledger.reserve_calls == reserved && tsg::g_test_file_reads == files_before
            && ledger.pending == 0 && ledger.valid
            && std::all_of(actual.begin(), actual.end(), [](float value) { return value == -999.0f; }),
            "Complete8192 device route refusal mutated/read/allocated/published before validation");
    };
    reject_complete_device_routes(false); reject_complete_device_routes(true);
    // fault=1 is an exact pre-dispatch wide-replacement error; fault=2 is
    // a deliberately injected file read error. Other exceptions fail the test.
    auto run = [&](int group, int rows, bool expected_success, int fault = 0) {
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
                std::vector<float> x(hidden), probability(top); std::vector<std::int32_t> selected(top);
                std::copy_n(input.begin() + row * hidden, hidden, x.begin());
                std::copy_n(routes.begin() + row * top, top, probability.begin());
                std::copy_n(ids.begin() + row * top, top, selected.begin());
                const auto answer = oracles[group].at(1)->compute_batch(x, selected, probability);
                std::copy(answer.begin(), answer.end(), expected.begin() + row * hidden);
            }
        } else expected = oracles[group].at(rows)->compute_batch(input, ids, routes);
        auto hm = weights[group].segment; hm.seq_len = rows; hm.independent_decode_rows = rows == 3;
        // Exercise the COMPLETE8192 routing download and device/output bridge
        // at the real maximum; smaller rows retain the existing host oracle.
        const bool device_bridge = rows == maximum_rows;
        if (device_bridge) {
            ggml_backend_tensor_set(widest_device.x, input.data(), 0, input.size() * sizeof(float));
            ggml_backend_tensor_set(widest_device.routes, routes.data(), 0, routes.size() * sizeof(float));
            ggml_backend_tensor_set(widest_device.output, actual.data(), 0, actual.size() * sizeof(float));
            hm.moe_in = widest_device.x; hm.weights = widest_device.routes; hm.moe_out = widest_device.output;
        }
        const auto before_execution = ExecutionStats();
        const auto before_files = tsg::g_test_file_reads, before_direct = tsg::g_test_direct_file_reads;
        int ok = -99; bool threw = false;
        try {
            ok = tsg::host_moe_cached_experts(hm, device_bridge ? nullptr : input.data(), ids.data(),
                device_bridge ? nullptr : routes.data(), device_bridge ? nullptr : actual.data(),
                rows > 8 ? "glm5-next bulk expert prefill" : "glm5-next selected expert seam", g_backend);
            if (device_bridge) {
                tsg::sync_backend(g_backend);
                ggml_backend_tensor_get(widest_device.output, actual.data(), 0, actual.size() * sizeof(float));
            }
        } catch (const std::runtime_error& error) {
            if (fault == 1 && std::strcmp(error.what(),
                "GLM wide scratch replacement refused/failed; persistent weights remain owned") == 0) { ok = 0; threw = true; }
            else if (fault == 2 && tsg::g_test_file_failure_countdown == 0
                && std::strcmp(error.what(), "Injected interrupted expert file read") == 0) { ok = 0; threw = true; }
            else throw;
        }
        if (direct_only_wide) {
            if (rows <= 8) {
                require(tsg::g_test_direct_file_reads == before_direct,
                    "Direct read experiment leaked into N1/N3 decode phase");
                narrow_file_reads += tsg::g_test_file_reads - before_files;
            } else wide_direct_reads += tsg::g_test_direct_file_reads - before_direct;
        }
        if (fault) require(threw && ExecutionStats().submissions == before_execution.submissions,
            "Injected phase failure did not abort before compute");
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
    using Snapshot = tsg::HostMoeCacheTestSnapshot;
    auto snapshot = [&](int group) {
        Snapshot value{};
        require(tsg::host_moe_expert_cache_test_lifecycle_snapshot(weights[group].segment, g_backend, value),
            "Expected current phase group has no live owner");
        return value;
    };
    auto stable_weights = [&](const Snapshot& a, const Snapshot& b) {
        require(a.entry_identity == b.entry_identity && a.weight_buffer == b.weight_buffer
            && a.weight_charge == b.weight_charge && a.narrow_buffer == b.narrow_buffer
            && a.narrow_charge == b.narrow_charge && a.weight_bytes == b.weight_bytes
            && a.narrow_bytes == b.narrow_bytes, "Narrow phase or reshape rebuilt affordable persistent weights");
    };
    if (coexist) {
        // The pool permits 1024 hot slots, while each complete current union
        // has only 64. The old near-global cold ceiling greedily fills hundreds
        // and evicts peers even though all three measured 64-slot plans fit.
        // Start at the measured maximum width: a smaller graph may legally
        // spend its scratch savings on optional slots, beyond this fixture's
        // 64-slot lifecycle snapshot. That is not a cache ownership failure.
        std::array<Snapshot, 3> original{};
        for (int group = 0; group < 3; ++group) {
            run(group, maximum_rows, true); original[group] = snapshot(group);
            require(original[group].capacity == count,
                "Optional cold slots consumed another affordable wide group's protected floor");
        }
        std::array<std::int64_t, 24> entries{}; int actual_entries = 0;
        require(TSGgml_HostMoeExpertCacheEntryStats(entries.data(), 3, &actual_entries) == 1 && actual_entries == 3,
            "Greedy cold wide planning evicted an otherwise affordable type group");
        const auto reserves = ledger.reserve_calls, releases = ledger.release_calls;
        const auto reads = tsg::g_test_file_reads;
        for (int rows : {3, 1, maximum_rows, 3, maximum_rows}) for (int group = 0; group < 3; ++group) {
            run(group, rows, true); stable_weights(original[group], snapshot(group));
        }
        require(ledger.reserve_calls == reserves && ledger.release_calls == releases
            && tsg::g_test_file_reads == reads,
            "Affordable whole unions were reallocated, evicted or reread during warm group switches");
        require(TSGgml_HostMoeExpertCacheEntryStats(entries.data(), 3, &actual_entries) == 1 && actual_entries == 3,
            "Warm phase transitions lost a coexisting type group");
        for (int rows : {128, 256, 384, 512, 1024}) for (int group = 0; group < 3; ++group) {
            run(group, rows, true); stable_weights(original[group], snapshot(group));
        }
        for (int group = 0; group < 3; ++group) original[group] = snapshot(group);
        // Tighten to the actual owned total after shrinking only the current
        // graph. Growing it again must reclaim an inactive group before the
        // injected allocation failure, rather than refunding virtual credit.
        run(0, 20, true); const auto small = snapshot(0);
        stable_weights(original[0], small);
        const auto tightened = Stats().reserved;
        require(tightened >= static_cast<std::int64_t>(plan.minimum())
            && tightened <= static_cast<std::int64_t>(budget)
            && original[0].wide_bytes > small.wide_bytes,
            "Fixture does not force real inactive-owner reclaim before wide growth");
        const auto before_tighten_releases = ledger.release_calls;
        require(TSGgml_UpdateHostMoeExpertCacheScope(token, tightened, &released) == 1
            && released == 0 && ledger.release_calls == before_tighten_releases,
            "Tightening to the actual owned total unnecessarily discarded a group");
        ledger.capacity = tightened;
        tsg::host_moe_expert_cache_test_fail_next(17);
        run(0, maximum_rows, false, 1);
        const auto failed = snapshot(0); stable_weights(small, failed);
        require(failed.wide_rows == 0 && failed.wide_bytes == 0
            && failed.wide_buffer == 0 && failed.wide_charge == 0,
            "Failed post-reclaim allocation must retain weights/narrow, not a stale wide graph");
        require(TSGgml_HostMoeExpertCacheEntryStats(entries.data(), 3, &actual_entries) == 1
            && actual_entries < 3 && ledger.release_calls > before_tighten_releases
            && Stats().reserved <= tightened && ledger.pending == 0 && ledger.valid,
            "Post-reclaim failure did not physically retire inactive owners or release its pending ticket");
        ledger.capacity = static_cast<std::int64_t>(budget);
        require(TSGgml_UpdateHostMoeExpertCacheScope(token, budget, &released) == 1,
            "Cannot restore an admitted phase budget after the injected failure");
        run(0, maximum_rows, true); stable_weights(small, snapshot(0));
        run(1, maximum_rows, true); run(2, maximum_rows, true);
    } else {
    // Actual cache retention: a large affordable entry is not reduced to its
    // fair cold-narrow quota merely because the current phase is N3/N1 decode.
    run(0, 128, true); const auto first = snapshot(0);
    const auto first_reserves = ledger.reserve_calls;
    run(0, 3, true); run(0, 1, true);
    stable_weights(first, snapshot(0));
    require(ledger.reserve_calls == first_reserves && snapshot(0).wide_rows == 128,
        "Decode invalidated the retained whole-union graph");
    // Cold sibling decode uses affordable spare capacity, not its optional fair
    // share if that would unnecessarily evict this still-affordable wide owner.
    run(1, 3, true); stable_weights(first, snapshot(0));
    // Complete min-1 gate precedes any eviction, ticket release or metadata
    // mutation. The same native scope resumes without reinstallation.
    const auto before_budget = Stats(); const auto before_resizes = ledger.release_calls;
    require(TSGgml_UpdateHostMoeExpertCacheScope(token, plan.minimum() - 1, &released) == 0
        && released == 0 && Stats().reserved == before_budget.reserved && Stats().budget == before_budget.budget
        && ledger.release_calls == before_resizes, "Sub-floor resize mutated the accepted phase state");
    stable_weights(first, snapshot(0)); run(0, 3, true);
    // A failed cold allocation is allowed to discard inactive optional cache,
    // but may not retain pending credit or publish a partial output.
    ledger.fail_commit = true; run(2, 128, false); ledger.fail_commit = false;
    require(ledger.pending == 0 && ledger.valid, "Failed phase commit stranded physical credit");
    run(1, 128, true); run(1, 3, true);
    run(2, 128, true); run(2, 3, true);
    run(0, 128, true); run(0, 3, true);
    // The admitted budget is below simultaneous-wide sum: at least one real
    // inactive owner must have been freed, not merely relabeled or discounted.
    require(ledger.release_calls > before_resizes && Stats().reserved <= static_cast<std::int64_t>(budget),
        "Wide phase did not perform real shared-owner retirement");
    for (int fault : {17, 18, 19}) {
        run(0, 128, true); const auto before = snapshot(0);
        tsg::host_moe_expert_cache_test_fail_next(fault);
        run(0, 64, false, 1);
        const auto after = snapshot(0); stable_weights(before, after);
        require(after.wide_bytes == 0 && after.wide_buffer == 0 && after.wide_charge == 0,
            "Failed reshape retained an uncommitted or duplicate wide buffer");
        run(0, 64, true); stable_weights(before, snapshot(0));
    }
    // Largest complete8192 device routing block and replacement ownership
    // stay correct at1024->512, including each physical/commit failure phase.
    for (int fault : {17, 18, 19}) {
        run(0, 1024, true); const auto before = snapshot(0);
        tsg::host_moe_expert_cache_test_fail_next(fault);
        run(0, 512, false, 1);
        const auto after = snapshot(0); stable_weights(before, after);
        require(after.wide_bytes == 0 && after.wide_buffer == 0 && after.wide_charge == 0
            && ledger.pending == 0 && ledger.valid, "Failed1024-to512 reshape retained duplicate or uncommitted ownership");
        run(0, 512, true); stable_weights(before, snapshot(0));
    }
    for (int rows : {20, 32, 128, 256, 384, 512, 1024, 8, 3, 1}) for (int group = 0; group < 3; ++group) run(group, rows, true);
    // Quiescent cancellation is represented by the real scope Trim operation,
    // not an asynchronous abort during a CUDA graph. Plan metadata survives.
    require(TSGgml_TrimHostMoeExpertCacheScope(token, 0, &released) == 1, "Phase cancellation trim failed");
    ledger.check_empty();
    require(TSGgml_UpdateHostMoeExpertCacheScope(token, budget + 1, &released) == 1
        && TSGgml_UpdateHostMoeExpertCacheScope(token, budget, &released) == 1,
        "Empty phase scope cannot resize/resume above its minimum");
    tsg::g_test_file_failure_countdown = 2;
    run(1, 1024, false, 2);
    require(ledger.valid && ledger.pending == 0, "Interrupted source read stranded a pending owner");
    run(1, 1024, true); run(1, 3, true);
    require(TSGgml_TrimHostMoeExpertCacheScope(token, 0, &released) == 1, "Cannot drain interrupted/resumed phase");
    ledger.check_empty();
    for (int group = 0; group < 3; ++group) run(group, 3, true);
    // Drop invalidates only referencing payload; the complete geometry plan
    // remains until backend/scope teardown. No source is silently re-registered.
    for (auto& bytes : weights[0].quantized) tsg::host_moe_expert_cache_on_drop(bytes.data());
    Snapshot dropped{};
    require(!tsg::host_moe_expert_cache_test_lifecycle_snapshot(weights[0].segment, g_backend, dropped),
        "Source drop retained its referencing payload");
    run(1, 3, true); run(2, 3, true);
    }
    require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1, "Three-group scope failed cleanup");
    token = 0; ledger.check_empty(); require(ledger.detach(), "Three-group callbacks retained an owner");
    for (auto& w : weights) for (auto& bytes : w.quantized) tsg::host_moe_expert_cache_on_drop(bytes.data());
    require(std::remove(path.c_str()) == 0, "Three-group file handles survived unregister");
    require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(budget, maximum_rows, 3, decode_rows, &token) == 1
        && tsg::host_moe_expert_cache_install_phase_plan(minima, maximum_rows, decode_rows, g_backend),
        "Ended scope leaked its phase metadata into a fresh session");
    require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1, "Fresh empty group scope did not close");
    token = 0;
    std::printf("PASS: actual phase coexist=%d minimum=%zu budget=%zu all-wide=%zu; %zu complete finite CUDA operator floats byte-exact, PP128/256/384/512/1024/N3/group switch, min-1 nonmutation, reshape faults, read interruption, trim/resume/drop/end. Not whole-model logits or asynchronous cancellation qualification.\n",
        coexist ? 1 : 0, plan.minimum(), budget, all_wide, compared);
    if (direct_only_wide) {
        if (!wide_direct_reads) std::puts("SKIP: wide phase completed buffered fallback but filesystem did not complete direct I/O");
        else std::printf("PASS: wide-only direct phase coexist=%d completed_direct_reads=%llu narrow_buffered_reads=%llu narrow_direct_reads=0; complete phase exact output/ownership/fault proof\n",
            coexist ? 1 : 0, static_cast<unsigned long long>(wide_direct_reads), static_cast<unsigned long long>(narrow_file_reads));
        require(coexist || narrow_file_reads > 0, "Wide-only direct fixture lacks cold N1/N3 buffered miss evidence");
    }
}
#endif
