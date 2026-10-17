// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// ============================================================================
// Key padding that lets ggml-metal run a long flash attention on its Metal 4
// tensor-API kernel.
//
// For a batch of queries ggml-metal has two flash-attention kernels. The
// simdgroup one takes 8 queries per threadgroup (OP_FLASH_ATTN_EXT_NQPSG, tuned
// for LLM decode), so a diffusion transformer's K/V are streamed again for every
// 8 queries. The tensor-API one takes 32 and runs on the M5's matrix units, but
// ggml_metal_op_flash_attn_ext_use_tensor (ggml-metal-ops.cpp) picks it only when
// the KEY count is a multiple of OP_FLASH_ATTN_EXT_TENSOR_NCPSG = 64
// (fa_tensor.metal: "the host guarantees ne11 % C == 0"). A DiT's key count is
// its packed sequence, set by the canvas and the clip length. MiniMax-H3 at
// 1280x768x107 is 34,991 tokens, 47 past a multiple of 64, so it never
// qualifies.
//
// Zero K/V rows up to the next multiple of 64, masked with -65504, leave the
// attention unchanged: a masked score's exp() is exactly 0, so it adds nothing
// to the softmax sum and its V row (zero) adds nothing to the output. -65504 is
// the largest finite half, so no step does Inf arithmetic. ggml-metal's block
// pre-pass (kernel_flash_attn_ext_blk) marks every all-zero 64-key block of the
// mask, and the kernel never reads the mask there.
//
// The mask needs a row per query (ggml-metal applies none to a query row past
// the mask's last), and at 34,991 tokens a [keys, queries] F16 mask is 2.45 GB.
// Every row is the same, though, so the queries go in as chunks of
// tsg_metal_flash_query_chunk rows: Q [hd, seq, heads] padded to whole chunks
// and reshaped to [hd, chunk, chunks * heads]. ggml's grouped-query broadcast
// sends Q head c + chunks * h to K/V head h, so one [keys, chunk] mask (4.3 MB
// at 34,991 tokens) serves every chunk of every head, and the kernel runs the
// same threadgroups over the same keys as the unchunked call.
//
// Measured on an M5 Pro against ggml a05c992, 56 heads x 128, F16 K/V, one
// call (eng/validation/metal-flash-kv-pad.cpp): 34,991 tokens 8.05 s -> 3.24 s
// (2.49x) chunked, 3.40 s with the full per-query mask; 9,298 tokens
// 0.456 -> 0.233 s (1.95x). Output equals the unpadded call bit for bit on CPU
// and to FP16 rounding on Metal. Without the tensor API the padded call is 5%
// SLOWER (0.457 -> 0.479 s), so the padding is gated on the tensor API being live.
//
// TS_METAL_FLASH_KV_PAD=0 turns the padding off, for A/B runs.
// ============================================================================

// True when `backend` is ggml-metal ("MTL0", "MTL1", ...).
inline bool tsg_backend_is_metal(ggml_backend_t backend)
{
    const char* name = backend != nullptr ? ggml_backend_name(backend) : nullptr;
    return name != nullptr && std::strncmp(name, "MTL", 3) == 0;
}

// Mirrors ggml-metal's has_tensor gating (ggml-metal-device.m) closely enough to
// predict whether the Metal 4 tensor API is live for this process: an
// M5/M6/A19/A20-class device name, or GGML_METAL_TENSOR_ENABLE, and no
// GGML_METAL_TENSOR_DISABLE. ggml does not publish the flag. A wrong guess costs
// speed, never correctness.
inline bool tsg_metal_tensor_api_likely(ggml_backend_t backend)
{
    if (!tsg_backend_is_metal(backend)) return false;
    if (std::getenv("GGML_METAL_TENSOR_DISABLE") != nullptr) return false;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    const char* desc = dev != nullptr ? ggml_backend_dev_description(dev) : nullptr;
    const bool m5class = desc != nullptr &&
        (std::strstr(desc, "M5") || std::strstr(desc, "M6") ||
         std::strstr(desc, "A19") || std::strstr(desc, "A20"));
    return m5class || std::getenv("GGML_METAL_TENSOR_ENABLE") != nullptr;
}

// The key count to hand ggml_flash_attn_ext: n_kv rounded up to a multiple of
// 64 when that is what keeps ggml-metal off its simdgroup kernel, else n_kv
// itself. `n_q` is the query count. Below 20 queries ggml-metal takes its vec
// kernel instead (ggml_metal_op_flash_attn_ext_use_vec), which needs no padding.
// The caller must also pass F16 or BF16 K and a head size of 64, 128, 192 (V
// 128), 256, 512 or 576 (V 512), which ggml-metal also requires.
inline int tsg_metal_flash_kv_length(ggml_backend_t backend, int n_kv, int n_q)
{
    constexpr int kTensorKeysPerBlock = 64;   // OP_FLASH_ATTN_EXT_TENSOR_NCPSG
    if (n_kv <= 0 || n_kv % kTensorKeysPerBlock == 0 || n_q < 20) return n_kv;
    static const bool enabled = [] {
        const char* e = std::getenv("TS_METAL_FLASH_KV_PAD");
        return e == nullptr || e[0] != '0';
    }();
    if (!enabled || !tsg_metal_tensor_api_likely(backend)) return n_kv;
    return (n_kv + kTensorKeysPerBlock - 1) / kTensorKeysPerBlock * kTensorKeysPerBlock;
}

// Query rows per chunk for the chunked call above: 64, doubled until the chunks
// of all heads fit one Metal grid dimension (65,535 threadgroups), so 64 up to
// 74,880 tokens at 56 heads. A multiple of the tensor kernel's 32 queries per
// threadgroup, so the chunking adds no partial query blocks.
inline int tsg_metal_flash_query_chunk(int n_q, int heads)
{
    int chunk = 64;
    while (static_cast<long long>((n_q + chunk - 1) / chunk) * heads > 65535) chunk *= 2;
    return chunk;
}

// The F16 [n_kv_padded, n_q] mask that hides keys [n_kv, n_kv_padded): 0 on a
// real key and -65504 on a pad, the same in every row. Built on the device from
// an arange, so nothing is uploaded; build it once per graph and share it.
inline ggml_tensor* tsg_metal_flash_kv_pad_mask(ggml_context* ctx, int n_kv, int n_kv_padded, int n_q)
{
    ggml_tensor* index = ggml_arange(ctx, 0.0f, static_cast<float>(n_kv_padded), 1.0f);
    // index - (n_kv - 1) clamped to [0, 1]: 0 for every real key, 1 for a pad.
    ggml_tensor* is_pad = ggml_clamp(ctx,
        ggml_scale_bias(ctx, index, 1.0f, static_cast<float>(1 - n_kv)), 0.0f, 1.0f);
    ggml_tensor* row = ggml_cast(ctx, ggml_scale(ctx, is_pad, -65504.0f), GGML_TYPE_F16);
    return ggml_repeat_4d(ctx, ggml_reshape_2d(ctx, row, n_kv_padded, 1), n_kv_padded, n_q, 1, 1);
}

// The chunked flash call. q: [hd, n_q, heads] F32 in any layout (the pad below
// copies it). k, v: [hd, n_kv_padded, heads] F16 with zero pad rows. mask: from
// tsg_metal_flash_kv_pad_mask(ctx, n_kv, n_kv_padded, chunk). Returns
// [hd * heads, n_q] -- the unchunked call's output, head-major per token -- or
// null when the backend cannot run the call.
inline ggml_tensor* tsg_metal_flash_attn_chunked(ggml_context* ctx, ggml_backend_t backend,
                                                 ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                                                 ggml_tensor* mask, int chunk, float scale)
{
    const std::int64_t hd = q->ne[0], n_q = q->ne[1], heads = q->ne[2];
    const std::int64_t chunks = (n_q + chunk - 1) / chunk;
    // Zero query rows up to whole chunks, written contiguously, then chunk c of
    // head h becomes Q head c + chunks * h, which ggml maps to K/V head h.
    ggml_tensor* qr = ggml_reshape_3d(ctx,
        ggml_pad(ctx, q, 0, static_cast<int>(chunks * chunk - n_q), 0, 0), hd, chunk, chunks * heads);
    ggml_tensor* out = ggml_flash_attn_ext(ctx, qr, k, v, mask, scale, 0.0f, 0.0f);
    if (out == nullptr || !ggml_backend_supports_op(backend, out)) return nullptr;
    ggml_prec_set_acc(out, GGML_PREC_F32);
    // out is [hd, chunks * heads, chunk] = [hd, chunks, heads, chunk]. Token
    // c * chunk + j sits at (c, j): permute to [hd, heads, chunk, chunks] and drop
    // the padded tail.
    ggml_tensor* tokens = ggml_cont(ctx,
        ggml_permute(ctx, ggml_reshape_4d(ctx, out, hd, chunks, heads, chunk), 0, 3, 1, 2));
    return ggml_view_2d(ctx, tokens, hd * heads, n_q, static_cast<std::size_t>(hd * heads) * sizeof(float), 0);
}
