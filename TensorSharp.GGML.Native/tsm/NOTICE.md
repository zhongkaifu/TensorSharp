# TensorSharp Metal kernels (tsm): third-party code

`kernels/quant.metal`, `kernels/units.metal` and `kernels/moe.metal` are copied unchanged from
[Strata](https://github.com/Niko1221/Strata) (`metal/kernels/`, commit `146df3c4`), MIT License,
Copyright (c) 2026 Niko1221 and the Strata contributors. The full license text is in
[`LICENSE-STRATA`](LICENSE-STRATA). Each file keeps a header naming its origin.

The rest of this directory is TensorSharp's own code under the repository's BSD-3-Clause license:
`tsm_moe.mm` / `tsm_moe.h` (the plan, the dispatch geometry, the integration with the tiered MoE seams on
ggml-metal's command queue), `kernels/tsm_moe_glue.metal` and `tsm_embed.cmake`. Two ideas in `tsm_moe.mm` follow
Strata's `metal/src/mtl_rt.mm` (same MIT terms): the seams' host turns on a shared event (`run_graph_seg`), and the
GPU keep-alive, whose spin kernel (`tsm_keepalive`) is Strata's `ka_spin` with its name changed.

The kernels also read the i-quant grids and block structs of ggml's `ggml-common.h` (MIT), which the build
prepends from the unmodified upstream checkout in `ExternalProjects/ggml`.
