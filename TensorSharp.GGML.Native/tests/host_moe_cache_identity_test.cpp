// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../ggml_ops_host_moe_cache_identity.h"
#include "../ggml_ops_expert_eviction.h"
#include <cstdio>
#include <stdexcept>

int main() {
    int checks = 0;
    auto require = [&](bool value) { ++checks; if (!value) throw std::runtime_error("Expert pool reused another source's bytes"); };
    char layer_a[3]{}, layer_b[3]{};
    std::array<const void*, 3> a{&layer_a[0], &layer_a[1], &layer_a[2]};
    std::array<const void*, 3> b{&layer_b[0], &layer_b[1], &layer_b[2]};
    tsg::ExpertSourceIdentity slot;
    require(!slot.matches(a, 0));
    slot.publish(a, 17);
    require(slot.matches(a, 17)); require(!slot.matches(a, 18)); require(!slot.matches(b, 17));
    for (int part = 0; part < 3; ++part) {
        require(slot.references(a[part])); require(!slot.references(b[part]));
        auto changed = a; changed[part] = b[part]; require(!slot.matches(changed, 17));
    }
    slot.invalidate();
    require(!slot.matches(a, 17)); require(!slot.references(a[0]));
    slot.publish(b, 17);
    require(slot.matches(b, 17)); require(!slot.matches(a, 17));
    // A model pool can contain more slots than one layer's local ID space.
    // Its metadata extent must authorize that capacity, even with plain LRU.
    tsg::ExpertEvictionPolicy pooled;
    pooled.configure(32, 24, false);
    require(pooled.epoch() == 0);
    require(pooled.prefer(31, 1, 17, 2));
    require(!pooled.prefer(31, 2, 17, 1));
    std::printf("PASS %d exact pooled source identity and interrupted-publication checks\n", checks);
}
