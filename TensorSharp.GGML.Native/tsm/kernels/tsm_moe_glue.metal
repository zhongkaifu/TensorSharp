// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// TensorSharp's glue around the grouped expert kernels of moe.metal (ported from Strata): the sum of the routes a
// tiered layer holds in its accelerator slots. moe_down has written ye[entry] = weight * (down . h[entry]) for every
// held route; a route the host computes has no entry (-1) and contributes nothing here.

// y[t][k] = sum over j of ye[ent[t][j]][k], j in route order (the order the routes were selected in).
kernel void tsm_moe_combine_held(device const float* ye [[buffer(0)]], device const int* ent [[buffer(1)]],
                                 device float* y [[buffer(2)]], constant uint4& a [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
    const uint n_embd = a.x, n_tok = a.y, topk = a.z;
    const uint t = gid / n_embd, k = gid % n_embd;
    if (t >= n_tok) return;
    float s = 0.0f;
    for (uint j = 0; j < topk; ++j) {
        const int e = ent[t * topk + j];
        if (e >= 0) s += ye[uint(e) * n_embd + k];
    }
    y[gid] = s;
}
