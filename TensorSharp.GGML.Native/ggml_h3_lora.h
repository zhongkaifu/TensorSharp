// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

// ============================================================================
// MiniMax-H3 DiT LoRA plug-ins (the LightX2V Turbo step-distillation adapters),
// applied UNMERGED inside TSGgml_MiniMaxH3DitForward: y = W x + up (down x). The
// quantized base weights are never modified; each update keeps its own F16/F32
// factors, as the reference does (inference_minimax_h3.py applies the adapter
// through PEFT and fuses only on request).
//
// Where the terms land (diffusers transformer_minimax_h3.py): attn.to_q/to_k/to_v
// on the q/k/v projections BEFORE the q/k RMSNorm and RoPE, attn.to_out.0 on
// out_proj, ff.net.0.proj on fc1 BEFORE SwiGLU and ff.net.2 on fc2, in the 50
// trunk blocks and the 2 token-refiner blocks alike.
//
// Layout that is easy to get wrong:
//   * The DiT's attn.qkv_proj is STACKED [q; k; v]: rows [0, inner) are q,
//     [inner, 2 inner) k and [2 inner, 3 inner) v. Every published single-file H3
//     checkpoint stores it that way, and diffusers splits it into to_q/to_k/to_v
//     by thirds (src/diffusers/loaders/single_file_utils.py:4319-4371,
//     convert_minimax_h3_transformer_checkpoint_to_diffusers). The per-head
//     INTERLEAVED to_qkv documented at ggml_ops_minimax_h3.cpp:27 is the video
//     VAE's and does NOT apply here. So the q/k/v terms are added to the three
//     thirds of the fused projection's output.
//   * mlp.fc1 fuses [gate; value] in the checkpoint, while diffusers' SwiGLU, and so
//     a Diffusers-format lora_B of ff.net.0.proj, reads [value; gate]
//     (convert_minimax_h3_gated_ff: "The checkpoint fuses [gate; value], SwiGLU
//     reads [value; gate]"). The managed loader swaps the two halves of up, so
//     fc1.up arrives here in fc1's own [gate | value] row order and the gate is
//     the first half, exactly as the base path's ggml_swiglu reads it.
//
// Memory at about 35k tokens: the fused path holds gv [2 ffn, seq] and the SwiGLU
// output, three [ffn, seq] tensors (6 GB). gv += delta would hold gv and a delta
// of the same size together (8 GB). h3_lora_fc1 below keeps the peak at three.
// out and fc2 add one [hidden, seq] delta each (0.75 GB); the q/k/v terms replace
// the slice copies they stand in for and add no full-size tensor.
// ============================================================================

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" {

// One projection's update, mirrored by H3Lora in GgmlNative.cs.
//   down: ggml [ne0 = in,   ne1 = rank] == PyTorch lora_A [rank, in]
//   up:   ggml [ne0 = rank, ne1 = out]  == lora_B [out, rank], with
//         strength * alpha / rank already folded in
//   type: GGML_TYPE_F32 (0) or GGML_TYPE_F16 (1), for both factors
//   rank: 0 = no update; otherwise the (padded) rank, the padding zero
struct TSGH3Lora
{
    void* down;
    void* up;
    std::int32_t type;
    std::int32_t rank;
    std::int64_t in, out;
};

// One block's updates. fc1.up is already in [gate | value] row order.
// Packing invariant the managed loader keeps and h3_lora_shrink exploits: when q,
// k and v all carry updates of one type, their downs sit back to back in one
// allocation (k.down == q.down + q.rank * q.in * element size, v likewise).
struct TSGH3BlockLora
{
    TSGH3Lora q, k, v, out, fc1, fc2;
};

// TSGgmlMiniMaxH3DitForwardDesc::adapter. struct_bytes is sizeof(TSGH3DitAdapter);
// the arrays hold at least as many entries as the forward runs blocks.
struct TSGH3DitAdapter
{
    std::int32_t struct_bytes, num_blocks, num_refiner_blocks, reserved;
    const TSGH3BlockLora* blocks;
    const TSGH3BlockLora* refiner;
};

} // extern "C"

// Pinned by MiniMaxH3NativeGraphAbiTests on the managed side.
static_assert(sizeof(void*) != 8 || (sizeof(TSGH3Lora) == 40 &&
    offsetof(TSGH3Lora, down) == 0 && offsetof(TSGH3Lora, up) == 8 &&
    offsetof(TSGH3Lora, type) == 16 && offsetof(TSGH3Lora, rank) == 20 &&
    offsetof(TSGH3Lora, in) == 24 && offsetof(TSGH3Lora, out) == 32),
    "MiniMax-H3 LoRA update must match the managed 64-bit ABI.");
static_assert(sizeof(void*) != 8 || sizeof(TSGH3BlockLora) == 240,
    "MiniMax-H3 LoRA block must match the managed 64-bit ABI.");
static_assert(sizeof(void*) != 8 || (sizeof(TSGH3DitAdapter) == 32 &&
    offsetof(TSGH3DitAdapter, blocks) == 16 && offsetof(TSGH3DitAdapter, refiner) == 24),
    "MiniMax-H3 LoRA adapter must match the managed 64-bit ABI.");

namespace tsg {

// Binds a graph tensor to caller-owned host bytes: the DiT's own `bind` (the
// device weight cache, else an input uploaded before compute).
using H3Bind = std::function<void(ggml_tensor*, void*, std::size_t)>;

// Upper bound on the nodes, and separately on the leaves, that one block's six
// updates add to the DiT graph. With every update present (eng/validation/
// minimax-h3-lora.cpp measures it):
//   q/k/v  one stacked shrink and its 3 row views, then per part an expand, the
//          slice view, the add and a reshape in place of view + cont      +10
//   out    shrink, expand, add                                             +3
//   fc1    x pre-scale, the shrink, 2 up row views, 2 expands, 2 weight row
//          views, 2 mul_mats, 2 rescales, 2 adds and swiglu_split in place
//          of pre-scale, mul_mat, rescale, swiglu                          +11
//   fc2    shrink, expand, add                                             +3
// 27 nodes (28 when silu + mul stands in for swiglu_split) and 10 factor
// leaves (4 downs, 6 ups); 48 keeps room for both.
constexpr int kH3LoraNodesPerBlock = 48;

inline bool h3_lora_has(const TSGH3Lora& l) { return l.rank > 0; }

inline bool h3_lora_has(const TSGH3BlockLora* b)
{
    return b != nullptr &&
        (h3_lora_has(b->q) || h3_lora_has(b->k) || h3_lora_has(b->v) ||
         h3_lora_has(b->out) || h3_lora_has(b->fc1) || h3_lora_has(b->fc2));
}

inline std::size_t h3_lora_down_bytes(const TSGH3Lora& l)
{
    return ggml_row_size(static_cast<ggml_type>(l.type), l.in) * static_cast<std::size_t>(l.rank);
}

inline std::size_t h3_lora_up_bytes(const TSGH3Lora& l)
{
    return ggml_row_size(static_cast<ggml_type>(l.type), l.rank) * static_cast<std::size_t>(l.out);
}

// factor * x with an F32 accumulator: ggml-cuda would otherwise run an F16
// factor's product in F16 (ggml-metal accumulates in F32 regardless).
inline ggml_tensor* h3_lora_product(ggml_context* ctx, ggml_tensor* factor, ggml_tensor* x)
{
    ggml_tensor* result = ggml_mul_mat(ctx, factor, x);
    ggml_prec_set_acc(result, GGML_PREC_F32);
    return result;
}

// The down projections of `parts` applied to one shared input x [in, seq]: a
// [rank, seq] tensor per part, null for a null part or one without an update.
// Parts whose downs sit back to back in one allocation (same type and `in`, as
// the loader packs a block's q/k/v) are bound as ONE [in, sum of ranks] tensor
// keyed by the first down and run as one mul_mat, so x is read once; each part
// is then a view of its rows. Ported from ggml_ops_qwen_image21.cpp
// (Builder::shrink), less its shared-factor case, which H3 never has.
inline std::vector<ggml_tensor*> h3_lora_shrink(ggml_context* ctx, const H3Bind& bind,
                                                std::initializer_list<const TSGH3Lora*> parts,
                                                ggml_tensor* x)
{
    std::vector<const TSGH3Lora*> list(parts);
    std::vector<ggml_tensor*> result(list.size(), nullptr);
    for (std::size_t i = 0; i < list.size();)
    {
        const TSGH3Lora* first = list[i];
        if (first == nullptr || !h3_lora_has(*first)) { ++i; continue; }
        if (x->ne[0] != first->in)
            throw std::invalid_argument("MiniMax-H3 LoRA: a down factor reads " +
                                        std::to_string(first->in) + " features but its input has " +
                                        std::to_string(x->ne[0]) + ".");
        // Extend the run while the next factor starts where this one ends.
        std::size_t end = i + 1;
        std::int64_t rank = first->rank;
        const char* next = static_cast<const char*>(first->down) + h3_lora_down_bytes(*first);
        while (end < list.size() && list[end] != nullptr && h3_lora_has(*list[end]) &&
               list[end]->type == first->type && list[end]->in == first->in &&
               static_cast<const char*>(list[end]->down) == next)
        {
            rank += list[end]->rank;
            next += h3_lora_down_bytes(*list[end]);
            ++end;
        }
        ggml_tensor* down = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(first->type), first->in, rank);
        bind(down, first->down, static_cast<std::size_t>(next - static_cast<const char*>(first->down)));
        ggml_tensor* t = h3_lora_product(ctx, down, x);
        std::size_t offset = 0;
        for (std::size_t j = i; j < end; ++j)
        {
            result[j] = end - i == 1 ? t :
                ggml_view_2d(ctx, t, list[j]->rank, t->ne[1], t->nb[1], offset * sizeof(float));
            offset += static_cast<std::size_t>(list[j]->rank);
        }
        i = end;
    }
    return result;
}

// up * shrunk: the update's [out, seq] term, with up [rank, out] bound at l.up.
inline ggml_tensor* h3_lora_expand(ggml_context* ctx, const H3Bind& bind, const TSGH3Lora& l,
                                   ggml_tensor* shrunk)
{
    if (shrunk == nullptr || shrunk->ne[0] != l.rank)
        throw std::invalid_argument("MiniMax-H3 LoRA: an up factor of rank " + std::to_string(l.rank) +
                                    " has no matching shrunk input.");
    ggml_tensor* up = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(l.type), l.rank, l.out);
    bind(up, l.up, h3_lora_up_bytes(l));
    return h3_lora_product(ctx, up, shrunk);
}

// delta + the [width, seq] slice of a packed projection at byte `offset`, as one
// contiguous [width, seq] tensor that stands in for the ggml_cont slicing the
// part out would need. The contiguous delta is src0: every backend's add takes
// the strided slice as its second operand, and the sum may reuse delta's memory.
inline ggml_tensor* h3_lora_add_slice(ggml_context* ctx, ggml_tensor* delta, ggml_tensor* packed,
                                      std::int64_t width, std::size_t offset)
{
    if (delta->ne[0] != width || delta->ne[1] != packed->ne[1])
        throw std::invalid_argument("MiniMax-H3 LoRA: an update's shape does not match its projection slice.");
    return ggml_add(ctx, delta, ggml_view_2d(ctx, packed, width, packed->ne[1], packed->nb[1], offset));
}

// fc1 with its update, as the SwiGLU halves (gate, value), each [ffn, seq].
//
// The base runs as two mul_mats over row views of the SAME bound fc1 weight (no
// second copy of it), the update as two expands over row views of up. Each half is
// its own subtree of the SwiGLU with the delta as its add's src[0], so
// ggml_build_forward_expand (src[0] first) runs gate delta, gate base, gate, value
// delta, value base, value. The gate is summed before the value starts, so at most
// three [ffn, seq] tensors are alive at once, as in the fused path (adding a full
// [2 ffn, seq] delta to gv would hold four). And the shrink, run first, is x's first
// consumer, which leaves the guard's pre-scale as its last: gallocr then scales x
// in place. Base first, the pre-scale needs its own [hidden, seq] whose hole the
// [ffn, seq] tensors cannot reuse, and the compute buffer at seq 2048 grew 1.11x
// over the fused path's (eng/validation/minimax-h3-lora.cpp measures it).
//
// guard_scale is h3_mm's activation pre-scale, applied when w1 is quantized: ONE
// x / guard_scale feeds both halves' base and each half is rescaled after its
// mul_mat, so a zero update reproduces the fused path's values. The shrink reads
// raw x; the guard exists for the q8_1 activation of a quantized weight, and the
// factors are F16/F32. fc1 must have no bias (the caller refuses an update on one
// that has).
inline std::pair<ggml_tensor*, ggml_tensor*> h3_lora_fc1(ggml_context* ctx, const H3Bind& bind,
                                                         ggml_tensor* w1, ggml_tensor* x,
                                                         const TSGH3Lora& l, std::int64_t ffn,
                                                         float guard_scale)
{
    if (w1->ne[0] != x->ne[0] || w1->ne[1] != 2 * ffn || l.out != 2 * ffn)
        throw std::invalid_argument("MiniMax-H3 LoRA: fc1 is not [in, 2 x ffn] or its update does not match it.");
    const bool guard = guard_scale > 0.0f && ggml_is_quantized(w1->type);
    ggml_tensor* xs = guard ? ggml_scale(ctx, x, 1.0f / guard_scale) : x;
    ggml_tensor* shrunk = h3_lora_shrink(ctx, bind, { &l }, x)[0];
    ggml_tensor* up = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(l.type), l.rank, l.out);
    bind(up, l.up, h3_lora_up_bytes(l));
    auto half = [&](std::int64_t row) {
        ggml_tensor* u = ggml_view_2d(ctx, up, l.rank, ffn, up->nb[1],
                                      static_cast<std::size_t>(row) * up->nb[1]);
        ggml_tensor* delta = h3_lora_product(ctx, u, shrunk);
        ggml_tensor* w = ggml_view_2d(ctx, w1, w1->ne[0], ffn, w1->nb[1],
                                      static_cast<std::size_t>(row) * w1->nb[1]);
        ggml_tensor* base = ggml_mul_mat(ctx, w, xs);
        if (guard) base = ggml_scale(ctx, base, guard_scale);
        return ggml_add(ctx, delta, base);
    };
    ggml_tensor* gate = half(0);
    ggml_tensor* value = half(ffn);
    return { gate, value };
}

// "" when `a` (null = no adapter) fits a forward of `num_blocks` trunk and
// `num_refiner` refiner blocks, else what is wrong, naming the block and the
// LoRA module (diffusers names) so the caller can say it in one line.
inline std::string h3_lora_validate(const TSGH3DitAdapter* a, int num_blocks, int num_refiner,
                                    std::int64_t hidden, std::int64_t inner, std::int64_t ffn)
{
    if (a == nullptr) return {};
    if (a->struct_bytes != static_cast<std::int32_t>(sizeof(TSGH3DitAdapter)))
        return "adapter is " + std::to_string(a->struct_bytes) + " bytes, this GgmlOps expects " +
               std::to_string(sizeof(TSGH3DitAdapter)) +
               "; the managed TensorSharp build and the native library differ, rebuild GgmlOps.";
    if (a->num_blocks < num_blocks)
        return "adapter covers " + std::to_string(a->num_blocks) + " blocks, but the forward runs " +
               std::to_string(num_blocks) + ".";
    if (a->num_refiner_blocks < num_refiner)
        return "adapter covers " + std::to_string(a->num_refiner_blocks) +
               " refiner blocks, but the forward runs " + std::to_string(num_refiner) + ".";
    if (num_blocks > 0 && a->blocks == nullptr) return "adapter has no block array.";
    if (num_refiner > 0 && a->refiner == nullptr) return "adapter has no refiner block array.";

    struct Part { const TSGH3Lora* l; const char* name; std::int64_t in, out; };
    auto check = [&](const TSGH3BlockLora& b, const std::string& where) -> std::string {
        const Part parts[] = {
            { &b.q, "attn.to_q", hidden, inner },
            { &b.k, "attn.to_k", hidden, inner },
            { &b.v, "attn.to_v", hidden, inner },
            { &b.out, "attn.to_out.0", inner, hidden },
            { &b.fc1, "ff.net.0.proj", hidden, 2 * ffn },
            { &b.fc2, "ff.net.2", ffn, hidden },
        };
        for (const Part& p : parts)
        {
            const TSGH3Lora& l = *p.l;
            if (l.rank == 0) continue;
            const std::string at = where + " " + p.name + ": ";
            if (l.rank < 0) return at + "rank " + std::to_string(l.rank) + " is negative.";
            if (l.type != GGML_TYPE_F32 && l.type != GGML_TYPE_F16)
                return at + "factor type " + std::to_string(l.type) + " is neither F32 (0) nor F16 (1).";
            if (l.down == nullptr || l.up == nullptr) return at + "a factor pointer is null.";
            if (l.in != p.in || l.out != p.out)
                return at + "factors are in " + std::to_string(l.in) + " x out " + std::to_string(l.out) +
                       ", but the layer is in " + std::to_string(p.in) + " x out " + std::to_string(p.out) + ".";
        }
        return {};
    };
    for (int i = 0; i < num_refiner; i++)
    {
        std::string bad = check(a->refiner[i], "refiner block " + std::to_string(i));
        if (!bad.empty()) return bad;
    }
    for (int i = 0; i < num_blocks; i++)
    {
        std::string bad = check(a->blocks[i], "block " + std::to_string(i));
        if (!bad.empty()) return bad;
    }
    return {};
}

} // namespace tsg
