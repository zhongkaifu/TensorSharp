// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../glm_batched_execution.h"
#include <cstdio>
#include <cstdlib>

int main() {
    int checks = 0;
    auto require = [&](bool value, const char* message) {
        ++checks;
        if (!value) { std::fprintf(stderr, "FAIL %s\n", message); std::exit(1); }
    };
    tsg_glm::batched_execution_recorder recorder;
    const std::int32_t slots[]{4, 9, 13}, before[]{17, 22, 27}, after[]{18, 23, 28};
    require(recorder.last().id == 0, "no planned execution masquerades as success");
    require(recorder.record_success(3, 765, 7, false, slots, before, after), "successful logical N3 graph");
    require(recorder.last().id == 1 && recorder.last().rows == 3 && recorder.last().nodes == 765, "accepted id and actual graph shape");
    require(recorder.last().backend_splits == 7 && !recorder.last().graph_reused, "backend dispatch splits are not launch count");
    for (int i = 0; i < 3; ++i)
        require(recorder.last().slots[i] == slots[i] && recorder.last().before[i] == before[i]
            && recorder.last().after[i] == after[i], "exact native slot head publication");
    recorder.begin_attempt();
    require(recorder.last().id == 0 && recorder.last().rows == 0, "failure before compute clears stale success");
    require(!recorder.failure_needs_reset(), "allocation or validation refusal is safe to retry");
    require(!recorder.begin_dispatch(17, slots) && !recorder.failure_needs_reset(), "invalid attempt cannot overrun row storage");
    require(!recorder.begin_dispatch(3, nullptr) && !recorder.failure_needs_reset(), "null dispatch does not start execution");
    require(recorder.begin_dispatch(3, slots) && recorder.failure_needs_reset(), "failed started compute requires reset");
    require(recorder.attempted_rows() == 3, "all rows participate in the failure fence");
    for (int i = 0; i < 3; ++i) require(recorder.attempted_slot(i) == slots[i], "failure fence has exact slot identities");
    require(!recorder.record_success(3, 765, 7, true, slots, before, before), "unadvanced state is not success");
    const std::int32_t duplicate[]{4, 4, 13};
    require(!recorder.record_success(3, 765, 7, true, duplicate, before, after), "aliased slots are not N3");
    require(!recorder.record_success(17, 765, 7, true, slots, before, after), "bounded evidence rows");
    require(!recorder.record_success(3, 0, 7, true, slots, before, after), "empty graph cannot be evidence");
    require(!recorder.record_success(3, 765, 0, true, slots, before, after), "missing backend execution cannot be evidence");
    require(!recorder.record_success(3, 765, 7, true, nullptr, before, after), "null slot identity is not success");
    require(recorder.last().id == 0, "invalid publication leaves no accepted evidence");
    require(recorder.failure_needs_reset(), "failed or partial publication remains unsafe to retry");
    require(recorder.record_success(3, 765, 7, true, slots, before, after), "success after refusal");
    require(recorder.last().id == 2 && recorder.last().graph_reused, "refusals do not increment accepted id");
    require(!recorder.failure_needs_reset(), "completed synchronized publication does not fence successful slots");
    recorder.begin_attempt();
    require(!recorder.failure_needs_reset() && recorder.last().id == 0, "next precompute attempt has an independent outcome");
    std::printf("PASS %d native evidence-recorder checks; no device/model qualification\n", checks);
}
