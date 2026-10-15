// Synthetic CUDA acceptance. Included inside the fixture's anonymous namespace.
// Complete output and exact physical owner lifetimes; no model-level perf claim.
#ifdef TSG_GGML_USE_CUDA
struct LifecycleLedger {
    struct Ticket { std::uint64_t token = 0; std::uintptr_t identity = 0; std::size_t bytes = 0, physical = 0; int role = -1; bool committed = false; };
    std::array<Ticket, 8> tickets{};
    std::size_t capacity = 0, host_capacity = 65536, pending = 0, committed = 0, host_pending = 0, host_committed = 0;
    std::size_t weight_bytes = 0, narrow_bytes = 0, weight_physical = 0, narrow_physical = 0;
    std::size_t peak_owned = 0, peak_physical = 0;
    std::uint64_t next = 0, reserve_calls = 0, commit_calls = 0, release_calls = 0;
    bool valid = true, attached = false, refuse_next_reserve = false, refuse_next_commit = false;
    Ticket* find(std::uint64_t token) noexcept { for (auto& t : tickets) if (t.token == token) return &t; return nullptr; }
    std::size_t role_physical(int role) noexcept {
        if (role == 3) return tsg::host_moe_expert_cache_test_file_workspace_bytes();
        std::size_t values[3]{}; tsg::host_moe_expert_cache_test_split_physical_bytes(values);
        return role >= 0 && role < 3 ? values[role] : 0;
    }
    void sample() noexcept {
        std::size_t values[3]{}; tsg::host_moe_expert_cache_test_split_physical_bytes(values);
        const auto total = values[0] + values[1] + values[2];
        if (total != tsg::host_moe_expert_cache_test_physical_bytes() || total > capacity) valid = false;
        peak_physical = (std::max)(peak_physical, total);
    }
    static std::uint64_t reserve(void* context, int rank, int kind, std::int64_t bytes) noexcept {
        auto& l = *static_cast<LifecycleLedger*>(context); ++l.reserve_calls;
        if (rank != 0 || bytes <= 0 || (kind != 4 && kind != 3)) { l.valid = false; return 0; }
        const bool host = kind == 3;
        int role = 3; std::uintptr_t identity = 0; std::size_t physical = 0;
        if (!host && (!tsg::host_moe_expert_cache_test_current_split_owner(role, identity, physical)
                || role < 0 || role > 2 || !identity || physical != 0)) { l.valid = false; return 0; }
        for (const auto& t : l.tickets) if (t.token && t.role == role) { l.valid = false; return 0; }
        if ((!host && role == 0 && static_cast<std::size_t>(bytes) != l.weight_bytes)
            || (!host && role == 1 && static_cast<std::size_t>(bytes) != l.narrow_bytes)) { l.valid = false; return 0; }
        if (!host && l.refuse_next_reserve) { l.refuse_next_reserve = false; return 0; }
        const auto owned = host ? l.host_pending + l.host_committed : l.pending + l.committed;
        const auto ceiling = host ? l.host_capacity : l.capacity;
        if (owned > ceiling || static_cast<std::size_t>(bytes) > ceiling - owned) return 0;
        Ticket* slot = nullptr; for (auto& t : l.tickets) if (!t.token) { slot = &t; break; }
        if (!slot) { l.valid = false; return 0; }
        const auto token = ++l.next; *slot = {token, identity, static_cast<std::size_t>(bytes), 0, role, false};
        (host ? l.host_pending : l.pending) += static_cast<std::size_t>(bytes);
        if (!host) l.peak_owned = (std::max)(l.peak_owned, l.pending + l.committed);
        l.sample(); return token;
    }
    static int commit(void* context, std::uint64_t token) noexcept {
        auto& l = *static_cast<LifecycleLedger*>(context); ++l.commit_calls;
        auto* t = l.find(token); if (!t || t->committed) { l.valid = false; return 0; }
        const bool host = t->role == 3;
        const auto physical = l.role_physical(t->role);
        if (!physical || physical > t->bytes) { l.valid = false; return 0; }
        if (!host) {
            int role = -1; std::uintptr_t identity = 0; std::size_t actual = 0;
            if (!tsg::host_moe_expert_cache_test_current_split_owner(role, identity, actual)
                || role != t->role || identity != t->identity || actual != physical) { l.valid = false; return 0; }
            const auto expected = role == 0 ? l.weight_physical : role == 1 ? l.narrow_physical : t->bytes;
            if (physical != expected) { l.valid = false; return 0; }
            if (l.refuse_next_commit) { l.refuse_next_commit = false; l.sample(); return 0; }
        }
        (host ? l.host_pending : l.pending) -= t->bytes;
        (host ? l.host_committed : l.committed) += t->bytes;
        t->committed = true; t->physical = physical; l.sample(); return 1;
    }
    static void release(void* context, std::uint64_t token) noexcept {
        auto& l = *static_cast<LifecycleLedger*>(context); ++l.release_calls;
        auto* t = l.find(token); if (!t) { l.valid = false; return; }
        const bool host = t->role == 3;
        if (l.role_physical(t->role) != 0) { l.valid = false; return; }
        if (!host) {
            int role = -1; std::uintptr_t identity = 0; std::size_t actual = 1;
            if (!tsg::host_moe_expert_cache_test_current_split_owner(role, identity, actual)
                || role != t->role || identity != t->identity || actual != 0) { l.valid = false; return; }
        }
        auto& owned = host ? (t->committed ? l.host_committed : l.host_pending) : (t->committed ? l.committed : l.pending);
        if (owned < t->bytes) { l.valid = false; return; }
        owned -= t->bytes; *t = {}; l.sample();
    }
    bool attach() {
        attached = tsg::SharedCacheCharge::attach(this, reserve, commit, release, false, true); return attached;
    }
    bool detach() {
        if (!tsg::SharedCacheCharge::detach(this)) return false; attached = false; return true;
    }
    void check_empty() {
        require(valid && !pending && !committed && !host_pending && !host_committed,
            "Lifecycle ledger retained or misattributed an owner");
        for (const auto& t : tickets) require(!t.token, "Lifecycle pending/committed ticket survived cleanup");
        std::size_t values[3]{}; tsg::host_moe_expert_cache_test_split_physical_bytes(values);
        require(!values[0] && !values[1] && !values[2] && !tsg::host_moe_expert_cache_test_physical_bytes()
            && !tsg::host_moe_expert_cache_test_file_workspace_bytes() && Stats().reserved == 0,
            "Lifecycle cleanup retained physical, host or private-accounted memory");
    }
    ~LifecycleLedger() {
        if (attached) { tsg::host_moe_expert_cache_release(); tsg::SharedCacheCharge::detach(this); }
    }
};

void glm_split_owner_lifecycle_checks(const std::array<ggml_type, 3>& types, float clamp) {
    static constexpr int count = 64, top = 8, max_rows = 32, decode_rows = 3;
    using Snapshot = tsg::HostMoeCacheTestSnapshot;
    using Plan = tsg::HostMoeCacheTestPlan;
    tsg::host_moe_expert_cache_release();
    Weights first, second; first.types = second.types = types;
    first.fill(.71f, count, top); second.fill(1.43f, count, top);
    first.segment.pooled_model_experts = second.segment.pooled_model_experts = count;
    first.segment.swiglu_clamp = second.segment.swiglu_clamp = clamp;
    second.segment.layer = 1;
    std::uint64_t token = 0;
    LifecycleLedger ledger;
    const auto path = fixture_file_path("glm-private-split-lifecycle-");
    struct Cleanup {
        std::string path; std::uint64_t& token;
        ~Cleanup() {
            tsg::g_test_file_failure_countdown = 0;
            tsg::host_moe_expert_cache_test_fail_next(0);
            tsg::host_moe_expert_cache_test_available_bytes((std::numeric_limits<std::size_t>::max)());
            std::int64_t released = 0;
            if (token) { TSGgml_EndHostMoeExpertCacheScope(token, &released); token = 0; }
            tsg::host_moe_expert_cache_release();
            tsg::host_moe_expert_cache_test_file_slice_limit(32u << 20);
            std::remove(path.c_str());
        }
    } cleanup{path, token};
    std::array<Weights*, 2> sources{&first, &second};
    std::int64_t offsets[2][3]{};
    {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        require(bool(file), "Cannot create split lifecycle source");
        for (int source = 0; source < 2; ++source) for (int part = 0; part < 3; ++part) {
            auto* weights = sources[source];
            file.write("source-epoch-offset!", 19);
            offsets[source][part] = static_cast<std::int64_t>(file.tellp());
            file.write(reinterpret_cast<const char*>(weights->quantized[part].data()), weights->quantized[part].size());
            require(bool(file), "Cannot write exact lifecycle source extent");
        }
    }
    for (int source = 0; source < 2; ++source) for (int part = 0; part < 3; ++part)
        require(TSGgml_RegisterHostFileSource(sources[source]->quantized[part].data(), sources[source]->quantized[part].size(),
            path.c_str(), offsets[source][part]) == 1, "Cannot register immutable lifecycle source key");
    std::array<std::map<int, std::unique_ptr<FullResidentGraph>>, 2> oracles;
    for (int source = 0; source < 2; ++source) for (int width : {1, 20, 21, 32})
        oracles[source].emplace(width, std::make_unique<FullResidentGraph>(*sources[source], width));
    // Independent references now own the original quantized bytes; mapped
    // identities are poisoned to make fallback or stale A->B source keys fail.
    for (auto* weights : sources) for (auto& bytes : weights->quantized) std::fill(bytes.begin(), bytes.end(), 0);
    DeviceInputs device(max_rows, top);
    std::array<Plan, 3> plans{};
    for (int i = 0; i < 3; ++i) {
        const int width = i == 0 ? 32 : i == 1 ? 20 : 21;
        require(tsg::host_moe_expert_cache_test_split_plan(first.segment, count, decode_rows, width, g_backend, plans[i]),
            "Cannot measure exact split owner plan without physical allocation");
        require(plans[i].capacity == count && plans[i].narrow_rows == 8 && plans[i].wide_rows == width
            && plans[i].weight_bytes > 0 && plans[i].narrow_bytes > 0 && plans[i].wide_bytes > 0
            && plans[i].weight_bytes == plans[i].weight_physical_bytes
            && plans[i].wide_bytes == plans[i].wide_physical_bytes
            && plans[i].narrow_bytes == plans[i].narrow_physical_bytes + (1u << 20),
            "Split plan conflates the three physical owners or helper/decode widths");
        if (i) require(plans[i].weight_bytes == plans[0].weight_bytes && plans[i].narrow_bytes == plans[0].narrow_bytes,
            "Wide-width planning rebuilt the persistent weights or narrow scratch");
    }
    auto total = [](const Plan& p) { return p.weight_bytes + p.narrow_bytes + p.wide_bytes; };
    const auto quota = (std::max)({total(plans[0]), total(plans[1]), total(plans[2])});
    ledger.capacity = quota; ledger.weight_bytes = plans[0].weight_bytes; ledger.narrow_bytes = plans[0].narrow_bytes;
    ledger.weight_physical = plans[0].weight_physical_bytes; ledger.narrow_physical = plans[0].narrow_physical_bytes;
    require(ledger.attach(), "Cannot attach three-owner lifecycle budget ledger");
    tsg::host_moe_expert_cache_test_file_slice_limit(65536);
    require(TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows(quota, max_rows, 1, decode_rows, &token) == 1,
        "Cannot begin split lifecycle scope with exactly three real decode rows");
    std::size_t minimum = 0;
    require(tsg::host_moe_expert_cache_minimum_prefill_plan(first.segment, decode_rows, max_rows, g_backend, minimum)
        && minimum == total(plans[0]) && !Stats().reserved && !tsg::host_moe_expert_cache_test_physical_bytes(),
        "Public admission minimum differs from the measured split physical owner plan");
    const auto before_mismatch = ledger.reserve_calls;
    std::size_t mismatched = 99;
    require(!tsg::host_moe_expert_cache_minimum_prefill_plan(first.segment, decode_rows, 16, g_backend, mismatched)
        && mismatched == 0 && !tsg::host_moe_expert_cache_scope_matches_plan(minimum, 1, 16, decode_rows)
        && ledger.reserve_calls == before_mismatch && !tsg::host_moe_expert_cache_test_physical_bytes(),
        "Scope32 accepted an incompatible native PP16 minimum or allocated during refusal");

    std::vector<float> input, routes, expected, actual;
    std::vector<std::int32_t> ids;
    auto prepare = [&](int rows, int source = 0) {
        input.resize(rows * hidden); routes.resize(rows * top); ids.resize(rows * top);
        expected.assign(rows * hidden, 0); actual.assign(rows * hidden, -999.f);
        for (int row = 0; row < rows; ++row) {
            for (int h = 0; h < hidden; ++h) input[row * hidden + h] = std::cos(h * .041f + row * .31f + source * .17f);
            for (int k = 0; k < top; ++k) {
                ids[row * top + k] = count - 1 - ((row % 8) * top + k);
                routes[row * top + k] = (k + 1.f) / 36.f;
            }
        }
        if (rows > 8) expected = oracles[source].at(rows)->compute_batch(input, ids, routes);
        else for (int row = 0; row < rows; ++row) {
            std::array<float, hidden> x{}; std::array<std::int32_t, top> selected{}; std::array<float, top> probabilities{};
            std::copy_n(input.data() + row * hidden, hidden, x.data());
            std::copy_n(ids.data() + row * top, top, selected.data());
            std::copy_n(routes.data() + row * top, top, probabilities.data());
            const auto answer = oracles[source].at(1)->compute(x, selected, probabilities);
            std::copy(answer.begin(), answer.end(), expected.begin() + row * hidden);
        }
        require(std::all_of(expected.begin(), expected.end(), [](float v) { return std::isfinite(v); })
            && std::any_of(expected.begin(), expected.end(), [](float v) { return v != 0.f; }),
            "Lifecycle ordinary CUDA reference is nonfinite or cannot distinguish poisoned fallback");
        ggml_backend_tensor_set(device.x, input.data(), 0, input.size() * sizeof(float));
        ggml_backend_tensor_set(device.routes, routes.data(), 0, routes.size() * sizeof(float));
        ggml_backend_tensor_set(device.output, actual.data(), 0, actual.size() * sizeof(float));
    };
    auto call = [&](int rows, int source = 0, int bridge = 0) {
        auto hm = sources[source]->segment; hm.seq_len = rows; hm.independent_decode_rows = rows <= decode_rows;
        hm.moe_in = device.x; hm.weights = device.routes; hm.moe_out = device.output;
        const auto result = tsg::host_moe_cached_experts(hm, bridge ? nullptr : input.data(), ids.data(),
            bridge ? nullptr : routes.data(), bridge == 2 ? nullptr : actual.data(),
            rows > 8 ? "glm5-next bulk expert prefill" : "glm5-next selected expert seam", g_backend);
        if (bridge == 2 && result == 1) {
            tsg::sync_backend(g_backend);
            ggml_backend_tensor_get(device.output, actual.data(), 0, actual.size() * sizeof(float));
        }
        return result;
    };
    auto exact = [&](int rows) {
        for (int row = 0; row < rows; ++row)
            require(std::memcmp(actual.data() + row * hidden, expected.data() + row * hidden, hidden * sizeof(float)) == 0,
                "A complete lifecycle row differs bitwise from its independent ordinary CUDA reference");
    };
    auto untouched = [&] {
        require(std::all_of(actual.begin(), actual.end(), [](float v) { return v == -999.f; }),
            "Refused lifecycle request published partial host output");
        std::vector<float> output(actual.size()); tsg::sync_backend(g_backend);
        ggml_backend_tensor_get(device.output, output.data(), 0, output.size() * sizeof(float));
        require(std::all_of(output.begin(), output.end(), [](float v) { return v == -999.f; }),
            "Refused lifecycle request published partial device output");
    };
    auto snapshot = [&](int source = 0) {
        Snapshot s{}; require(tsg::host_moe_expert_cache_test_lifecycle_snapshot(sources[source]->segment, g_backend, s),
            "Missing actual split-owner cache snapshot"); return s;
    };
    auto payload = [&](int source = 0) {
        const auto s = snapshot(source); std::array<std::vector<std::uint8_t>, 3> bytes;
        for (int part = 0; part < 3; ++part) {
            bytes[part].resize(s.weight_part_bytes[part]);
            require(!bytes[part].empty() && tsg::host_moe_expert_cache_test_copy_weight_payload(sources[source]->segment,
                g_backend, part, bytes[part].data(), bytes[part].size()), "Cannot capture the real persistent weight payload");
        }
        return bytes;
    };
    auto owners = [&](const Snapshot& s) {
        std::size_t values[3]{}; tsg::host_moe_expert_cache_test_split_physical_bytes(values);
        require(ledger.valid && s.capacity == count && s.narrow_rows == 8 && s.weight_buffer && s.narrow_buffer
            && s.weight_charge && s.narrow_charge && s.weight_bytes == ledger.weight_bytes && s.narrow_bytes == ledger.narrow_bytes
            && s.weight_physical_bytes == values[0] && s.narrow_physical_bytes == values[1] && s.wide_physical_bytes == values[2]
            && s.weight_bytes + s.narrow_bytes + s.wide_bytes == static_cast<std::size_t>(Stats().reserved)
            && s.weight_bytes + s.narrow_bytes + s.wide_bytes == ledger.committed && !ledger.pending
            && s.committed_owners == (s.wide_bytes ? 3 : 2) && ledger.host_committed == 65536 && !ledger.host_pending,
            "Live split ownership differs from actual physical buffers and immutable tickets");
        require((s.wide_bytes != 0) == (s.wide_buffer != 0 && s.wide_charge != 0 && s.wide_rows > 8),
            "Wide scratch owner has phantom or uncharged physical storage");
    };
    auto stable_weights = [&](const Snapshot& before, const Snapshot& after) {
        require(before.entry_identity == after.entry_identity && before.weight_buffer == after.weight_buffer
            && before.narrow_buffer == after.narrow_buffer && before.weight_charge == after.weight_charge
            && before.narrow_charge == after.narrow_charge && before.weight_bytes == after.weight_bytes
            && before.narrow_bytes == after.narrow_bytes && before.growth_generation == after.growth_generation,
            "A scratch reshape rebuilt persistent weights, narrow scratch or their tickets");
        for (int part = 0; part < 3; ++part)
            require(before.weight_data[part] == after.weight_data[part] && before.weight_part_bytes[part] == after.weight_part_bytes[part],
                "A scratch reshape moved the actual quantized weight payload");
    };
    auto keys = [&](const Snapshot& s, int source = 0) {
        std::array<bool, count> seen{}; int residents = 0;
        for (int part = 0; part < 3; ++part)
            require(s.source[part] == reinterpret_cast<std::uintptr_t>(sources[source]->quantized[part].data()),
                "Pooled source identity does not name the current immutable layer");
        for (int slot = 0; slot < s.capacity; ++slot) {
            const auto& key = s.slots[slot]; require(key.expert >= 0 && key.expert < count && !seen[key.expert],
                "Resident lifecycle table lost or duplicated a local expert ID"); seen[key.expert] = true; ++residents;
            for (int part = 0; part < 3; ++part) require(key.source[part] == s.source[part],
                "Resident expert retained a stale immutable layer source key");
            require(key.last_used > 0 && key.last_used <= s.clock, "Resident lifecycle LRU age is invalid");
        }
        require(residents == count && s.resident == count, "Warm lifecycle union is not fully resident");
    };
    auto no_state_change = [&](const Snapshot& before, const Snapshot& after) {
        stable_weights(before, after);
        require(before.clock == after.clock && before.resident == after.resident && before.wide_generation == after.wide_generation,
            "Refused scratch replacement changed hot cache clock or resident generation");
        for (int slot = 0; slot < count; ++slot) {
            require(before.slots[slot].expert == after.slots[slot].expert && before.slots[slot].last_used == after.slots[slot].last_used,
                "Refused scratch replacement invalidated an expert or LRU age");
            for (int part = 0; part < 3; ++part) require(before.slots[slot].source[part] == after.slots[slot].source[part],
                "Refused scratch replacement changed a resident immutable source key");
        }
    };
    std::int64_t released = 0;
    prepare(32);
    require(TSGgml_UpdateHostMoeExpertCacheScope(token, minimum - 1, &released) == 1, "Cannot set split min-1 quota");
    const auto before_minimum = ledger.reserve_calls;
    require(call(32) == 0 && ledger.reserve_calls == before_minimum, "Split min-1 allocated a ticket or dispatched");
    untouched(); ledger.check_empty();
    require(TSGgml_UpdateHostMoeExpertCacheScope(token, quota, &released) == 1, "Cannot restore measured split quota");
    ledger.capacity = minimum - 1;
    require(call(32) == 0, "Split shared min-1 did not refuse the complete owner set");
    untouched(); ledger.check_empty(); ledger.capacity = quota;
    const auto before_physical = ledger.reserve_calls;
    tsg::host_moe_expert_cache_test_available_bytes((std::size_t(512) << 20) + minimum - 1);
    require(call(32) == 0 && ledger.reserve_calls == before_physical, "Split physical min-1 reached shared reserve or dispatch");
    untouched(); ledger.check_empty();
    tsg::host_moe_expert_cache_test_available_bytes((std::numeric_limits<std::size_t>::max)());
    for (int fault : {1, 2, 3, 11, 12, 13, 14, 15, 16}) {
        tsg::host_moe_expert_cache_test_fail_next(fault);
        require(call(32) == 0, "A split cold physical/commit fault did not fail closed");
        untouched(); ledger.check_empty();
    }

    // Capacity preparation is not padded execution: first live N1 allocates
    // the measured scope32 resources but submits exactly one N1 graph.
    prepare(1); const auto cold_execution = ExecutionStats();
    require(call(1, 0, 2) == 1, "Cold actual N1 cannot prepare the measured scope32 owners"); exact(1);
    require_operator_batch_shape(1, top);
    const auto cold_one = snapshot(); owners(cold_one);
    require(cold_one.capacity == count && cold_one.wide_rows == max_rows && cold_one.committed_owners == 3
        && cold_one.resident == top && cold_one.weight_bytes + cold_one.narrow_bytes + cold_one.wide_bytes == total(plans[0])
        && ExecutionStats().submissions == cold_execution.submissions + 1 && ExecutionStats().joint == cold_execution.joint,
        "Cold N1 shrank the full64 table, omitted upfront wide32, or executed fake rows");

    Snapshot previous{}; std::array<std::vector<std::uint8_t>, 3> original_payload;
    bool populated = false;
    for (int rows : {32, 20, 21, 32, 3, 1}) for (int bridge = 0; bridge < 3; ++bridge) {
        prepare(rows); const auto reads = tsg::g_test_file_reads; const auto misses = Stats().misses;
        const auto execution = ExecutionStats();
        require(call(rows, 0, bridge) == 1, "Actual split lifecycle width did not engage"); exact(rows);
        require_operator_batch_shape(rows, top);
        const auto after = snapshot(); owners(after); keys(after);
        const auto completed = ExecutionStats();
        require(completed.submissions == execution.submissions + 1
            && completed.joint == execution.joint + (rows == 3 ? 1 : 0),
            "Actual lifecycle rows ran scalar submissions or misreported concurrent decode");
        if (!populated) {
            stable_weights(cold_one, after);
            require(after.wide_generation == cold_one.wide_generation,
                "First wide32 rebuilt the cold N1 prepared scope32 generation");
            original_payload = payload(); populated = true;
        }
        else {
            stable_weights(previous, after);
            require(tsg::g_test_file_reads == reads && Stats().misses == misses && payload() == original_payload,
                "All-hit scratch reshape reread source or replaced persistent quantized bytes");
            require(after.clock == previous.clock + static_cast<std::uint64_t>(rows * top),
                "Scratch reshape reset or skipped the real LRU access clock");
            std::array<bool, count> routed{}; for (const auto id : ids) routed[id] = true;
            for (int slot = 0; slot < count; ++slot) {
                require(after.slots[slot].expert == previous.slots[slot].expert, "Scratch reshape changed resident slot assignment");
                require(routed[after.slots[slot].expert] ? after.slots[slot].last_used > previous.slots[slot].last_used
                    : after.slots[slot].last_used == previous.slots[slot].last_used,
                    "Scratch reshape lost a routed/unrouted expert's actual LRU age");
            }
        }
        if (rows > 8) require(after.wide_rows == rows && after.wide_bytes > 0,
            "Wide lifecycle executed padded or stale-width scratch");
        std::printf("LIFECYCLE rows=%d bridge=%d requests=%d resident=%d owners=%d wide=%d clock=%llu reads=%llu W=%zu N=%zu B=%zu\n",
            rows, bridge, rows == 3 ? 3 : 1, after.resident, after.committed_owners, after.wide_rows,
            static_cast<unsigned long long>(after.clock), static_cast<unsigned long long>(tsg::g_test_file_reads),
            after.weight_physical_bytes, after.narrow_physical_bytes, after.wide_physical_bytes);
        previous = after;
    }
    auto warm_refusal = [&](int rows, int fault, bool reserve_refusal, bool commit_refusal) {
        prepare(rows); const auto before = snapshot(); const auto reads = tsg::g_test_file_reads;
        const auto execution = ExecutionStats(); const auto misses = Stats().misses;
        ledger.refuse_next_reserve = reserve_refusal; ledger.refuse_next_commit = commit_refusal;
        if (fault) tsg::host_moe_expert_cache_test_fail_next(fault);
        int refused = -99; bool exact_native_refusal = false;
        try { refused = call(rows); }
        catch (const std::runtime_error& error) {
            // Accept only this exact intended pre-dispatch native request error.
            // Assertion, numeric, I/O, and unrelated failures must escape the test.
            if (std::strcmp(error.what(), "GLM wide scratch replacement refused/failed; persistent weights remain owned") != 0) throw;
            exact_native_refusal = true; refused = 0;
        }
        require(refused == 0 && exact_native_refusal && !ledger.refuse_next_reserve && !ledger.refuse_next_commit,
            "Warm split replacement did not exercise/refuse its exact owner stage"); untouched();
        const auto failed = snapshot(); owners(failed); no_state_change(before, failed); keys(failed);
        require(!failed.wide_bytes && !failed.wide_physical_bytes && failed.committed_owners == 2
            && tsg::g_test_file_reads == reads && Stats().misses == misses
            && ExecutionStats().submissions == execution.submissions && payload() == original_payload,
            "Refused scratch retained duplicate storage, touched experts or published output");
        require(call(rows, 0, 2) == 1, "Failed split replacement could not recover"); exact(rows);
        const auto recovered = snapshot(); owners(recovered); keys(recovered); stable_weights(before, recovered);
        require(recovered.wide_rows == rows && recovered.wide_generation > before.wide_generation
            && tsg::g_test_file_reads == reads && payload() == original_payload,
            "Scratch recovery lost the retained weight generation or warm file locality");
    };
    warm_refusal(20, 0, true, false);
    warm_refusal(21, 0, false, true);
    warm_refusal(32, 17, false, false);
    warm_refusal(20, 18, false, false);
    warm_refusal(21, 19, false, false);

    // New immutable layer keys with equal layout and local IDs must never hit A.
    // Interrupt the first source group before all three payload parts publish.
    prepare(32, 1); const auto before_source = snapshot(); const auto before_source_execution = ExecutionStats();
    tsg::g_test_file_failure_countdown = 2; bool interrupted = false;
    try { call(32, 1); } catch (const std::runtime_error&) { interrupted = true; }
    require(interrupted && tsg::g_test_file_failure_countdown == 0
        && ExecutionStats().submissions == before_source_execution.submissions,
        "Interrupted source epoch submitted a graph or silently fell back"); untouched();
    const auto incomplete = snapshot(1); owners(incomplete); stable_weights(before_source, incomplete);
    for (int slot = 0; slot < count; ++slot) if (incomplete.slots[slot].expert >= 0)
        require(incomplete.slots[slot].source[0] != reinterpret_cast<std::uintptr_t>(second.quantized[0].data()),
            "Interrupted first expert group published an incomplete new source key");
    require(call(32, 1, 2) == 1, "Interrupted source group did not drain and recover"); exact(32);
    const auto source_ready = snapshot(1); owners(source_ready); keys(source_ready, 1); stable_weights(before_source, source_ready);
    const auto second_payload = payload(1); require(second_payload != original_payload,
        "New layer keys reused stale original quantized expert payload");
    for (int rows : {20, 3, 1}) {
        prepare(rows, 1); const auto reads = tsg::g_test_file_reads;
        require(call(rows, 1, 2) == 1, "Recovered immutable source cannot execute a warm shape"); exact(rows);
        const auto after = snapshot(1); owners(after); keys(after, 1); stable_weights(source_ready, after);
        require(tsg::g_test_file_reads == reads && payload(1) == second_payload,
            "Recovered source epoch lost warm weights during a scratch reshape");
    }
    require(ledger.peak_owned <= quota && ledger.peak_physical <= quota,
        "Split lifecycle overlapped old/new weight or scratch physical generations");
    require(TSGgml_EndHostMoeExpertCacheScope(token, &released) == 1, "Cannot end split lifecycle ownership scope");
    token = 0; ledger.check_empty(); require(ledger.detach(), "Split lifecycle callbacks survived complete cleanup");
    tsg::host_moe_expert_cache_release();
    require(std::remove(path.c_str()) == 0, "Immutable source handles survived complete split lifecycle teardown");
    std::printf("PASS: Native18 split-owner coldN1(upfrontW64+wide32)->N32->20->21->32->N3->N1, K8/64experts, 3 bridges, exact ordinary-width CUDA rows, preserved pointers/payload/keys/LRU/generation, no warm file reads, min-1/shared/physical, cold1/2/3/11..16, warm reserve/commit/17..19, interrupted source A->B/recovery and all owners free-before-credit; clamp=%g quota=%zu peak_owned=%zu peak_physical=%zu\n",
        clamp, quota, ledger.peak_owned, ledger.peak_physical);
}
#endif
