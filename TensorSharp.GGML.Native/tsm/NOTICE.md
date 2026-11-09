# TensorSharp Metal kernels (tsm): third-party code

`kernels/quant.metal`, `kernels/units.metal` and `kernels/moe.metal` are copied unchanged from
[Strata](https://github.com/Niko1221/Strata) (`metal/kernels/`, commit `146df3c4`), MIT License,
Copyright (c) 2026 Niko1221 and the Strata contributors. The full license text is in
[`LICENSE-STRATA`](LICENSE-STRATA). Each file keeps a header naming its origin.

`kernels/tsm_gdn_prompt.metal` is ported from Strata's `metal/kernels/gdn_prompt.metal` (same commit and terms):
`gdn_pre`, `gdn_scan_r<4, 16>` and `gdn_post`, changed to read and write the recurrent state in ggml's gated delta net
layout, to use ggml's L2 norm (`x / max(|x|, eps)`), to write fp32 only, and to stage the conv history's new tail
(`tsm_gdn_hist_store`); its header lists the changes. `kernels/tsm_hc_prompt.metal` (the hyper-connection mix and
scatter passes) is TensorSharp's, written after Strata's `gr_epi_up` and `gr_norm_h` (`metal/kernels/prompt.metal`).
`kernels/tsm_moe_mm.metal` is ported from Strata's `metal/kernels/moe_mm.metal` (same commit and terms): the prompt
chunk's grouped expert GEMM on the Metal 4 tensor API (`mm_hist`, `mm_scan`, `mm_place`, `mm_clear`, `mm_gather`,
`mm_gu3`, `mm_down3`, `mm_combine`) with the Q2_0 decode, trimmed to those kernels and their Q2_0 instantiations; the
kernel bodies are unchanged and its header lists what was left out.
`kernels/tsm_qsa_prompt.metal` (the QSA cell selection and attention) is TensorSharp's own; `kernels/tsm_qsa_attn_mm.metal`
(the attention on the Metal 4 tensor API) is TensorSharp's, structured after ggml's `kernels/fa_tensor.metal` (MIT) with
the query heads as rows as in Strata's `qsa_attn_mm`.

The rest of this directory is TensorSharp's own code under the repository's BSD-3-Clause license:
`tsm_moe.mm` / `tsm_moe.h` (the plan, the dispatch geometry, the integration with the tiered MoE seams and the
prompt-width device operations on ggml-metal's command queue), `kernels/tsm_moe_glue.metal` and `tsm_embed.cmake`. Two ideas in `tsm_moe.mm` follow
Strata's `metal/src/mtl_rt.mm` (same MIT terms): the seams' host turns on a shared event (`run_graph_seg`), and the
GPU keep-alive, whose spin kernel (`tsm_keepalive`) is Strata's `ka_spin` with its name changed.

The kernels also read the i-quant grids and block structs of ggml's `ggml-common.h` (MIT), which the build
prepends from the unmodified upstream checkout in `ExternalProjects/ggml`.
