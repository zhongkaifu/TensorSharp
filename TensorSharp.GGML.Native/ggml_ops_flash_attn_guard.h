// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>

// ============================================================================
// Flash attention that never reaches a backend without a kernel for its shape.
//
// ggml_backend_graph_compute does not consult ggml_backend_supports_op: a node
// the backend has no kernel for is simply executed. For GGML_OP_FLASH_ATTN_EXT
// on CUDA that is a process abort - fattn.cu's ggml_cuda_flash_attn_ext hits
// `case BEST_FATTN_KERNEL_NONE: GGML_ABORT("fatal error")`. At ggml 456172ec,
// ggml_cuda_get_best_fattn_kernel returns NONE when:
//
//   * the K head size is not 40/64/72/80/96/112/128/256 (V equal), or one of
//     192 (V 128), 320 (V 256), 512 (V 512), 576 (V 512) - e.g. the head size
//     16/32 of the synthetic test models, or any odd head size;
//   * the head size is 192/320/512/576 and the grouped-query optimisation does
//     not apply: that needs gqa_ratio >= 2, a mask, max_bias == 0, K->ne[1] a
//     multiple of FATTN_KQ_STRIDE (256) and every nb[1..3] of the unquantized
//     Q/K/V/mask divisible by 16 (plus gqa_ratio % 8 for 192, % 32 for 320).
//     A Gemma 4 global layer (head 512) whose cache is smaller than 256 rows,
//     or a window clamped to a cache length that is not a multiple of 256,
//     lands here;
//   * K or V is not F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0;
//   * the mask has ne[2] != 1.
//
// ggml_backend_supports_op reports exactly that predicate (it calls the same
// ggml_cuda_get_best_fattn_kernel), and Metal/Vulkan report their own head-size
// and type limits the same way, so the check here is the backend's own answer
// for the exact node, not a copy of its rules.
//
// tsg_flash_attn_ext_guarded builds ggml_flash_attn_ext and returns it when the
// backend can run it. Otherwise it returns the same attention written as
// explicit ops - F32 mul_mat, soft_max_ext (mask, scale, ALiBi, sinks, logit
// softcap), mul_mat - with the flash node's output shape [Dv, H, N, B] and
// contiguity, and reports the site once on stderr. It is never silent.
//
// Q: [D, N, H, B] F32. K: [D, S, Hk, Bk], V: [Dv, S, Hk, Bk] of any K/V type
// the backend can copy to F32. H % Hk == 0, B % Bk == 0. mask: F16
// [S, Npad >= N, 1 or H, 1 or B] or null. sinks: F32 [H] or null.
// ============================================================================

// The explicit attention alone (no support check, no warning). Query rows are
// processed in chunks whose [S, rows, H, B] F32 score matrix stays within
// `score_budget_bytes` (0 = the default 256 MiB), so a long prefill that falls
// back never allocates O(N * S) at once. Exposed for the native test.
ggml_tensor* tsg_attention_explicit(
    ggml_context* ctx,
    ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* mask,
    float scale, float max_bias, float logit_softcap, ggml_tensor* sinks,
    std::int64_t score_budget_bytes = 0);

// `site` is a short static label naming the call site in the warning.
// `backend` null means "no backend to ask" and always returns the flash node.
// `prec` is applied to the flash node when it is not GGML_PREC_DEFAULT (the
// explicit path always multiplies in F32).
ggml_tensor* tsg_flash_attn_ext_guarded(
    ggml_context* ctx, ggml_backend_t backend, const char* site,
    ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* mask,
    float scale, float max_bias, float logit_softcap,
    ggml_tensor* sinks = nullptr, ggml_prec prec = GGML_PREC_DEFAULT);

// Number of graph builds that took the explicit path since process start.
std::uint64_t tsg_flash_attn_fallback_count();
